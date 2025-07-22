import torch
import random


class RandomColorTempTint(torch.nn.Module):
    """
    이미지의 색온도와 틴트를 랜덤하게 조절하는 커스텀 Transform.
    입력 텐서는 [0, 1] 범위로 정규화되어 있다고 가정합니다.
    """

    def __init__(self, temp_strength=0.1, tint_strength=0.1):
        super().__init__()
        # 조절할 최대 강도를 저장합니다.
        self.temp_strength = temp_strength
        self.tint_strength = tint_strength

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image (Tensor): (..., C, H, W) 형태의 이미지 텐서
        Returns:
            Tensor: 색온도/틴트가 조절된 이미지 텐서
        """
        # -strength ~ +strength 범위에서 랜덤한 값을 뽑습니다.
        temp_shift = random.uniform(-self.temp_strength, self.temp_strength)
        tint_shift = random.uniform(-self.tint_strength, self.tint_strength)

        # 채널 분리 (C, H, W) 또는 (B, C, H, W) 모두 처리 가능
        r, g, b = image.chunk(3, dim=-3)

        # 색온도 조절: R 채널과 B 채널을 반대로 조절
        r = r + temp_shift
        b = b - temp_shift

        # 틴트 조절: G 채널 조절
        g = g + tint_shift

        # 다시 채널 합치기
        image = torch.cat([r, g, b], dim=-3)

        # 픽셀 값이 [0, 1] 범위를 벗어나지 않도록 클램핑
        image = torch.clamp(image, 0, 1)

        return image