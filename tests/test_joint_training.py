"""CPU integration checks for joint gradients and encoder checkpoint restoration."""

import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image
from torch import nn

# These tests exercise the actual token helpers with a tiny encoder. The
# production OpenCLIP factory is replaced, so no pretrained download is needed.
if importlib.util.find_spec("open_clip") is None:
    sys.modules["open_clip"] = types.ModuleType("open_clip")
    try:
        from person_search import training
    finally:
        del sys.modules["open_clip"]
else:
    from person_search import training


class TinyTransformer(nn.Module):
    batch_first = True

    def __init__(self):
        super().__init__()
        self.layer = nn.TransformerEncoderLayer(8, 2, 16, dropout=0, batch_first=True)

    def forward(self, x, attn_mask=None):
        return self.layer(x, src_mask=attn_mask)


class TinyCLIP(nn.Module):
    def __init__(self):
        super().__init__()
        self.visual = nn.Module()
        self.visual.conv1 = nn.Conv2d(3, 8, 2, stride=2)
        self.visual.class_embedding = nn.Parameter(torch.randn(8))
        self.visual.positional_embedding = nn.Parameter(torch.randn(5, 8))
        self.visual.ln_pre = nn.LayerNorm(8)
        self.visual.transformer = TinyTransformer()
        self.visual.ln_post = nn.LayerNorm(8)
        self.visual.proj = nn.Parameter(torch.eye(8))
        self.token_embedding = nn.Embedding(16, 8)
        self.positional_embedding = nn.Parameter(torch.randn(6, 8))
        self.transformer = TinyTransformer()
        self.ln_final = nn.LayerNorm(8)
        self.text_projection = nn.Parameter(torch.eye(8))
        self.training_modes = []

    def train(self, mode=True):
        self.training_modes.append(mode)
        return super().train(mode)


def preprocess(image):
    pixels = torch.tensor(list(image.getdata()), dtype=torch.float32)
    return pixels.reshape(4, 4, 3).permute(2, 0, 1) / 255


def tokenizer(texts):
    return torch.tensor([[1, 2 + len(text) % 10, 15, 0, 0, 0] for text in texts])


class JointTrainingTest(unittest.TestCase):
    def test_joint_training_and_evaluation_restore_same_encoder(self):
        torch.set_num_threads(1)
        torch.manual_seed(9)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "imgs").mkdir()
            records = []
            for split in ("train", "val", "test"):
                for pid in range(2):
                    name = f"{split}-{pid}.png"
                    Image.new("RGB", (4, 4), (50 + pid * 100, 80, 120)).save(root / "imgs" / name)
                    records.append({"id": pid, "split": split, "img_path": name,
                                    "captions": ["red coat" if pid else "blue trousers"]})
            (root / "data_captions.json").write_text(json.dumps(records))
            encoder = TinyCLIP()
            before = {k: v.detach().clone() for k, v in encoder.state_dict().items()}
            checkpoint = root / "joint.pth"
            with patch.object(training, "load_model", return_value=(encoder, preprocess, tokenizer, "cpu")):
                training.train_reranker(
                    root, root / "initial.pth", checkpoint, device="cpu", batch_size=2,
                    epochs=1, encoder_learning_rate=1e-3, model_name="tiny", image_size=(4, 4),
                )
            self.assertTrue(all(p.requires_grad for p in encoder.parameters()))
            self.assertTrue(torch.isfinite(encoder.visual.conv1.weight.grad).all())
            self.assertTrue(torch.isfinite(encoder.token_embedding.weight.grad).all())
            self.assertGreater(encoder.visual.conv1.weight.grad.abs().sum().item(), 0)
            self.assertGreater(encoder.token_embedding.weight.grad.abs().sum().item(), 0)
            self.assertFalse(torch.equal(before["visual.conv1.weight"], encoder.visual.conv1.weight))
            self.assertFalse(torch.equal(before["token_embedding.weight"], encoder.token_embedding.weight))
            self.assertIn(True, encoder.training_modes)
            self.assertFalse(encoder.training)
            payload = torch.load(checkpoint, weights_only=False)
            self.assertTrue(payload["encoder_finetuned"])
            for key, value in encoder.state_dict().items():
                torch.testing.assert_close(payload["encoder_state_dict"][key], value)

            restored = TinyCLIP()
            def global_features(model, *args):
                self.assertFalse(torch.is_grad_enabled())
                self.assertFalse(model.training)
                for key, value in model.state_dict().items():
                    torch.testing.assert_close(value, payload["encoder_state_dict"][key])
                return torch.eye(2)

            with patch.object(training, "load_model", return_value=(restored, preprocess, tokenizer, "cpu")), \
                 patch.object(training, "encode_images", side_effect=global_features), \
                 patch.object(training, "encode_text", side_effect=global_features):
                report = training.evaluate_reranked(
                    root, root / "initial.pth", checkpoint, device="cpu", batch_size=2,
                    top_m=2, model_name="tiny", image_size=(4, 4),
                )
            self.assertEqual(report["encoder_source"], "reranker_checkpoint")
            self.assertEqual(report["baseline"]["R@1"], 100)

            # Old frozen checkpoints must still evaluate without encoder weights.
            payload.pop("encoder_state_dict")
            payload.pop("encoder_finetuned")
            torch.save(payload, checkpoint)
            with patch.object(training, "load_model", return_value=(encoder, preprocess, tokenizer, "cpu")), \
                 patch.object(training, "encode_images", return_value=torch.eye(2)), \
                 patch.object(training, "encode_text", return_value=torch.eye(2)):
                report = training.evaluate_reranked(
                    root, root / "initial.pth", checkpoint, device="cpu", batch_size=2,
                    top_m=2, model_name="tiny", image_size=(4, 4),
                )
            self.assertEqual(report["encoder_source"], "clip_checkpoint")


if __name__ == "__main__":
    unittest.main()
