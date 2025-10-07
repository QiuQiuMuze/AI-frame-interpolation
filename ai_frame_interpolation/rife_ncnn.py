"""Wrappers around the rife-ncnn-vulkan command line interpolator."""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


@dataclass
class RifeNcnnConfig:
    """Configuration for the rife-ncnn-vulkan binary.

    Attributes:
        binary: Path to the rife-ncnn-vulkan executable.
        model: Optional model name shipped with the binary (for example
            "rife-v4.6"). If ``None`` the binary default is used.
        scale: Image scale factor, see upstream documentation.
        tta: Enables test-time augmentation.
        uhd: Enables UHD mode for very high resolution inputs.
        gpu_id: GPU index to execute on. ``None`` leaves the upstream default.
        num_threads: Number of CPU threads to allow. ``None`` leaves upstream
            default.
    """

    binary: Path
    model: Optional[str] = None
    scale: float = 1.0
    tta: bool = False
    uhd: bool = False
    gpu_id: Optional[int] = None
    num_threads: Optional[int] = None

    def validate(self) -> None:
        """Validate the configuration and raise ``ValueError`` when invalid."""

        if self.scale <= 0:
            raise ValueError("scale must be positive")
        if self.binary is None:
            raise ValueError("binary path must be provided")
        if not shutil.which(str(self.binary)) and not Path(self.binary).exists():
            raise FileNotFoundError(
                f"rife-ncnn-vulkan executable not found at {self.binary}."
            )


class RifeNcnnInterpolator:
    """Run the rife-ncnn-vulkan binary for image pair interpolation."""

    def __init__(self, config: RifeNcnnConfig) -> None:
        self._config = config
        self._config.validate()

    def build_command(
        self,
        first_image: Path,
        second_image: Path,
        output: Path,
        time: float = 0.5,
        extra_args: Optional[Iterable[str]] = None,
    ) -> list[str]:
        """Create the rife-ncnn-vulkan command line invocation."""

        if not 0.0 <= time <= 1.0:
            raise ValueError("time must be within [0, 1]")

        cmd = [str(self._config.binary)]
        cmd += ["-0", str(first_image)]
        cmd += ["-1", str(second_image)]
        cmd += ["-o", str(output)]
        cmd += ["-t", f"{time:.6f}"]

        if self._config.model:
            cmd += ["-m", self._config.model]
        if self._config.scale != 1.0:
            cmd += ["-s", f"{self._config.scale:.4f}"]
        if self._config.tta:
            cmd.append("-x")
        if self._config.uhd:
            cmd.append("-u")
        if self._config.gpu_id is not None:
            cmd += ["-g", str(self._config.gpu_id)]
        if self._config.num_threads is not None:
            cmd += ["-j", str(self._config.num_threads)]

        if extra_args:
            cmd.extend(extra_args)

        return cmd

    def interpolate(
        self,
        first_image: Path,
        second_image: Path,
        output: Path,
        time: float = 0.5,
        extra_args: Optional[Iterable[str]] = None,
        check: bool = True,
    ) -> subprocess.CompletedProcess:
        """Execute interpolation and return the completed process."""

        command = self.build_command(
            first_image=first_image,
            second_image=second_image,
            output=output,
            time=time,
            extra_args=extra_args,
        )

        return subprocess.run(command, check=check)


__all__ = ["RifeNcnnConfig", "RifeNcnnInterpolator"]
