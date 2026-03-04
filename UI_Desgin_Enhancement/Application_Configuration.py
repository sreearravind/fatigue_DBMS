@dataclass(frozen=True)
class AppConfig:
    APP_TITLE: str = "Fatigue Data Intelligence Dashboard"
    APP_ICON: str = "🔬"  # Changed from 🧠 to microscope for metallurgy theme
    APP_LOGO_TEXT: str = "⚙️ Metallurgical Fatigue Analysis Platform"
    APP_VERSION: str = "v2.0"
    RANDOM_SEED: int = 42
    COMPANY_BRANDING: str = "Materials Intelligence Lab"
    
    # Color scheme for consistent theming
    COLORS: Dict = field(default_factory=lambda: {
        "primary": "#0066CC",
        "success": "#28A745", 
        "warning": "#FFC107",
        "danger": "#DC3545",
        "info": "#17A2B8",
        "dark": "#343A40",
        "light": "#F8F9FA"
    })
