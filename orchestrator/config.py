"""Configuration management for orchestrator."""

import yaml
from typing import Dict, List, Optional
from dataclasses import dataclass
from pathlib import Path


@dataclass
class MemberConfig:
    """Configuration for a council member."""
    id: str
    url: str
    temperature: float
    node: str
    gpu: int


@dataclass
class ChairmanConfig:
    """Configuration for the chairman."""
    url: str
    node: str
    gpus: List[int]


@dataclass
class TimeoutConfig:
    """Timeout configuration."""
    generation: int = 120
    judging: int = 60
    chairman: int = 180


class OrchestratorConfig:
    """Main configuration class for orchestrator."""
    
    def __init__(
        self,
        models_config_path: str = "./config/models.yaml",
        endpoints_config_path: str = "./config/endpoints.yaml",
    ):
        """
        Initialize configuration.
        
        Args:
            models_config_path: Path to models configuration
            endpoints_config_path: Path to endpoints configuration
        """
        self.models_config_path = Path(models_config_path)
        self.endpoints_config_path = Path(endpoints_config_path)
        
        self.members: List[MemberConfig] = []
        self.chairman: Optional[ChairmanConfig] = None
        self.timeouts: TimeoutConfig = TimeoutConfig()
        
        self._load_config()
    
    def _load_config(self):
        """Load configuration from files."""
        # Load endpoints config
        with open(self.endpoints_config_path, "r") as f:
            endpoints_data = yaml.safe_load(f)
        
        # Parse members
        for member_data in endpoints_data.get("members", []):
            member = MemberConfig(
                id=member_data["id"],
                url=member_data["url"],
                temperature=member_data["temperature"],
                node=member_data["node"],
                gpu=member_data["gpu"],
            )
            self.members.append(member)
        
        # Parse chairman
        chairman_data = endpoints_data.get("chairman", {})
        if chairman_data:
            self.chairman = ChairmanConfig(
                url=chairman_data["url"],
                node=chairman_data["node"],
                gpus=chairman_data["gpus"],
            )
        
        # Parse timeouts
        timeout_data = endpoints_data.get("timeouts", {})
        if timeout_data:
            self.timeouts = TimeoutConfig(
                generation=timeout_data.get("generation", 120),
                judging=timeout_data.get("judging", 60),
                chairman=timeout_data.get("chairman", 180),
            )
    
    def get_member_by_id(self, member_id: str) -> Optional[MemberConfig]:
        """Get member configuration by ID."""
        for member in self.members:
            if member.id == member_id:
                return member
        return None
    
    def get_all_member_urls(self) -> List[str]:
        """Get all member URLs."""
        return [member.url for member in self.members]
    
    def get_chairman_url(self) -> str:
        """Get chairman URL."""
        if self.chairman:
            return self.chairman.url
        raise ValueError("Chairman not configured")
    
    def __repr__(self) -> str:
        return (
            f"OrchestratorConfig(members={len(self.members)}, "
            f"chairman={'configured' if self.chairman else 'not configured'})"
        )

