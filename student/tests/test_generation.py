"""CPU tensor tests; executed on Colab where torch is already installed."""
from types import SimpleNamespace

import pytest
from PIL import Image

torch = pytest.importorskip("torch")

from student.inference import generate_batch  # noqa: E402


def test_batch_images_left_padding_and_output_trimming(tmp_path):
    Image.new("RGB", (640, 360)).save(tmp_path / "a.jpg")
    rows = [{"image_id": str(i), "image_path": "a.jpg", "width": 640, "height": 360}
            for i in range(2)]
    input_ids = torch.tensor([[0, 1] + [7] * 220, [1, 1] + [7] * 220])
    class Batch(dict):
        def to(self, device):
            assert device == "cpu"
            return self
    class Processor:
        def apply_chat_template(self, conversations, **kwargs):
            assert len(conversations) == 2
            assert all(isinstance(c[0]["content"][0]["image"], Image.Image) for c in conversations)
            assert kwargs["padding"] and kwargs["add_generation_prompt"]
            return Batch(input_ids=input_ids,
                         attention_mask=(input_ids != 0).long(),
                         token_type_ids=torch.zeros_like(input_ids),
                         mm_token_type_ids=torch.ones_like(input_ids))
        def batch_decode(self, tokens, **kwargs):
            assert tokens.tolist() == [[8, 9], [10, 11]]
            return ["[]", "[]"]
    class Model:
        device = "cpu"
        config = SimpleNamespace(image_token_id=7)
        def generate(self, **kwargs):
            assert "token_type_ids" not in kwargs
            assert "mm_token_type_ids" in kwargs
            assert kwargs["do_sample"] is False
            return torch.cat([kwargs["input_ids"], torch.tensor([[8, 9], [10, 11]])], dim=1)
    assert generate_batch(Model(), Processor(), rows, tmp_path) == ["[]", "[]"]


def test_unexpected_image_token_count_fails_before_generation(tmp_path):
    Image.new("RGB", (640, 360)).save(tmp_path / "a.jpg")
    row = {"image_id": "a", "image_path": "a.jpg", "width": 640, "height": 360}
    processor = SimpleNamespace(apply_chat_template=lambda *args, **kwargs:
                                {"input_ids": torch.tensor([[7, 7]])})
    model = SimpleNamespace(config=SimpleNamespace(image_token_id=7))
    with pytest.raises(ValueError, match="220 image tokens"):
        generate_batch(model, processor, [row], tmp_path)
