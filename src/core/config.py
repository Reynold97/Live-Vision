# src/core/config.py
import os
import logging
import json
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from pydantic import Field, ConfigDict, model_validator
from pydantic.dataclasses import dataclass

# Load environment variables
load_dotenv()

@dataclass
class APISettings:
    """API server settings."""
    HOST: str = Field(default="0.0.0.0", description="API host")
    PORT: int = Field(default=8000, description="API port")
    DEBUG: bool = Field(default=False, description="Debug mode")
    CORS_ORIGINS: List[str] = Field(default_factory=lambda: ["*"], description="CORS origins")
    ENVIRONMENT: str = Field(default="development", description="Environment (development, production, testing)")
    
    @model_validator(mode='after')
    def validate_environment(self) -> 'APISettings':
        allowed = ['development', 'production', 'testing']
        if self.ENVIRONMENT.lower() not in allowed:
            raise ValueError(f"Environment must be one of: {allowed}")
        self.ENVIRONMENT = self.ENVIRONMENT.lower()
        return self

@dataclass
class PipelineSettings:
    """Pipeline configuration."""
    BASE_DATA_DIR: str = Field(default="data", description="Base directory for data")
    MAX_CONCURRENT_PIPELINES: int = Field(default=5, description="Maximum concurrent pipelines")
    MIN_CHUNK_DURATION: int = Field(default=10, description="Minimum chunk duration in seconds")
    MAX_CHUNK_DURATION: int = Field(default=300, description="Maximum chunk duration in seconds")
    DEFAULT_CHUNK_DURATION: int = Field(default=30, description="Default chunk duration in seconds")
    # Update supported source types to include m3u8
    SUPPORTED_SOURCE_TYPES: List[str] = Field(default_factory=lambda: ["youtube", "m3u8"], 
                                       description="Supported source types")
    DEFAULT_RUNTIME_DURATION: int = Field(default=-1, description="Default runtime duration in minutes, -1 for indefinite")
    
    @model_validator(mode='after')
    def validate_settings(self) -> 'PipelineSettings':
        if self.MAX_CONCURRENT_PIPELINES < 1:
            raise ValueError("MAX_CONCURRENT_PIPELINES must be at least 1")
            
        if self.MIN_CHUNK_DURATION < 5:
            raise ValueError("MIN_CHUNK_DURATION must be at least 5 seconds")
                
        if self.MAX_CHUNK_DURATION > 600:
            raise ValueError("MAX_CHUNK_DURATION cannot exceed 600 seconds")
                
        if self.DEFAULT_CHUNK_DURATION < self.MIN_CHUNK_DURATION or self.DEFAULT_CHUNK_DURATION > self.MAX_CHUNK_DURATION:
            raise ValueError(f"DEFAULT_CHUNK_DURATION must be between {self.MIN_CHUNK_DURATION} and {self.MAX_CHUNK_DURATION}")
                    
        return self

@dataclass
class AnalysisSettings:
    """Analysis configuration."""
    GOOGLE_API_KEY: str = Field(default="", description="Google API Key for Gemini")
    DEFAULT_ANALYSIS_PROMPT: str = Field(default="""
        Analyze this video segment and focus on identifying commercial elements and opportunities. Please:

        1. First describe the main content and context of this segment:
        - What type of event or content is this?
        - Who or what is being shown?
        - What is the setting or location?

        2. Identify specific commercial elements:
        - Teams, athletes, or performers involved
        - Venues or locations
        - Equipment or gear being used
        - Brands or logos visible
        - Any merchandise already being displayed
        - Ads

        3. Search the web and provide:
        - Official merchandise stores for any teams/performers identified
        - Related products with direct purchase links
        - Licensed merchandise availability
        - Similar products or alternatives

        Format your response in markdown with clear sections and include specific URLs where available. 
        Keep your analysis focused on legitimate, official merchandise and commercial opportunities.
        If you cannot identify specific commercial elements, provide context about the general category of products related to the content.
        Don't make any introduction or conclusions, just straight to the points.
    """, description="Default prompt for analysis")
    USE_WEB_SEARCH: bool = Field(default=True, description="Use web search by default")
    MAX_RETRIES: int = Field(default=3, description="Maximum retries for API calls")
    RETRY_DELAY: int = Field(default=15, description="Delay between retries in seconds")
    EXPORT_RESPONSES: bool = Field(default=True, description="Export full responses to files for analysis")
    
    @model_validator(mode='after')
    def validate_api_key(self) -> 'AnalysisSettings':
        if not self.GOOGLE_API_KEY and os.getenv('ENVIRONMENT', 'development') == 'production':
            raise ValueError("GOOGLE_API_KEY is required in production")
        return self

@dataclass
class LoggingSettings:
    """Logging configuration."""
    LOG_LEVEL: str = Field(default="INFO", description="Log level")
    LOG_FORMAT: str = Field(default="%(asctime)s - %(name)s - %(levelname)s - %(message)s", description="Log format")
    LOG_DIR: str = Field(default="logs", description="Log directory")
    USE_JSON_LOGS: bool = Field(default=False, description="Use JSON logging format")
    
    @model_validator(mode='after')
    def validate_log_level(self) -> 'LoggingSettings':
        allowed = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if self.LOG_LEVEL.upper() not in allowed:
            raise ValueError(f"LOG_LEVEL must be one of: {allowed}")
        self.LOG_LEVEL = self.LOG_LEVEL.upper()
        return self

@dataclass
class Settings:
    """Combined application settings."""
    API: APISettings = Field(default_factory=APISettings)
    PIPELINE: PipelineSettings = Field(default_factory=PipelineSettings)
    ANALYSIS: AnalysisSettings = Field(default_factory=AnalysisSettings)
    LOGGING: LoggingSettings = Field(default_factory=LoggingSettings)
    
    def __post_init__(self) -> None:
        """Apply environment variables after initialization."""
        self._apply_env_vars()
    
    def _apply_env_vars(self) -> None:
        """Apply environment variables to override default settings."""
        # Override API settings
        self.API.HOST = os.getenv("API_HOST", self.API.HOST)
        self.API.PORT = int(os.getenv("API_PORT", str(self.API.PORT)))
        self.API.DEBUG = os.getenv("API_DEBUG", str(self.API.DEBUG)).lower() in ('true', '1', 'yes')
        self.API.ENVIRONMENT = os.getenv("ENVIRONMENT", self.API.ENVIRONMENT)
        
        cors_origins = os.getenv("CORS_ORIGINS")
        if cors_origins:
            self.API.CORS_ORIGINS = cors_origins.split(",")
        
        # Override Pipeline settings
        self.PIPELINE.BASE_DATA_DIR = os.getenv("BASE_DATA_DIR", self.PIPELINE.BASE_DATA_DIR)
        self.PIPELINE.MAX_CONCURRENT_PIPELINES = int(os.getenv("MAX_CONCURRENT_PIPELINES", 
                                                             str(self.PIPELINE.MAX_CONCURRENT_PIPELINES)))
        self.PIPELINE.MIN_CHUNK_DURATION = int(os.getenv("MIN_CHUNK_DURATION", 
                                                       str(self.PIPELINE.MIN_CHUNK_DURATION)))
        self.PIPELINE.MAX_CHUNK_DURATION = int(os.getenv("MAX_CHUNK_DURATION", 
                                                       str(self.PIPELINE.MAX_CHUNK_DURATION)))
        self.PIPELINE.DEFAULT_CHUNK_DURATION = int(os.getenv("DEFAULT_CHUNK_DURATION", 
                                                           str(self.PIPELINE.DEFAULT_CHUNK_DURATION)))
        
        source_types = os.getenv("SUPPORTED_SOURCE_TYPES")
        if source_types:
            self.PIPELINE.SUPPORTED_SOURCE_TYPES = source_types.split(",")
        
        # Override Analysis settings
        self.ANALYSIS.GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", self.ANALYSIS.GOOGLE_API_KEY)
        self.ANALYSIS.USE_WEB_SEARCH = os.getenv("USE_WEB_SEARCH", 
                                             str(self.ANALYSIS.USE_WEB_SEARCH)).lower() in ('true', '1', 'yes')
        self.ANALYSIS.MAX_RETRIES = int(os.getenv("MAX_RETRIES", str(self.ANALYSIS.MAX_RETRIES)))
        self.ANALYSIS.RETRY_DELAY = int(os.getenv("RETRY_DELAY", str(self.ANALYSIS.RETRY_DELAY)))
        self.ANALYSIS.EXPORT_RESPONSES = os.getenv("EXPORT_RESPONSES", 
                                      str(self.ANALYSIS.EXPORT_RESPONSES)).lower() in ('true', '1', 'yes')
        
        custom_prompt = os.getenv("DEFAULT_ANALYSIS_PROMPT")
        if custom_prompt:
            self.ANALYSIS.DEFAULT_ANALYSIS_PROMPT = custom_prompt
        
        # Override Logging settings
        self.LOGGING.LOG_LEVEL = os.getenv("LOG_LEVEL", self.LOGGING.LOG_LEVEL)
        self.LOGGING.LOG_DIR = os.getenv("LOG_DIR", self.LOGGING.LOG_DIR)
        self.LOGGING.USE_JSON_LOGS = os.getenv("USE_JSON_LOGS", 
                                          str(self.LOGGING.USE_JSON_LOGS)).lower() in ('true', '1', 'yes')

# Create a global settings object
settings = Settings()

def get_settings() -> Settings:
    """Get application settings."""
    return settings

def initialize_logger():
    """Initialize the application logger."""
    import logging
    import sys
    import json
    from logging.handlers import RotatingFileHandler
    import os
    
    # Create logs directory if it doesn't exist
    os.makedirs(settings.LOGGING.LOG_DIR, exist_ok=True)
    
    # Get the root logger
    root_logger = logging.getLogger()
    
    # Clear existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Set log level
    log_level = getattr(logging, settings.LOGGING.LOG_LEVEL)
    root_logger.setLevel(log_level)
    
    # Create formatters
    if settings.LOGGING.USE_JSON_LOGS:
        class JsonFormatter(logging.Formatter):
            def format(self, record):
                log_record = {
                    "timestamp": self.formatTime(record, self.datefmt),
                    "name": record.name,
                    "level": record.levelname,
                    "message": record.getMessage(),
                }
                
                if record.exc_info:
                    log_record["exception"] = self.formatException(record.exc_info)
                    
                return json.dumps(log_record)
                
        formatter = JsonFormatter()
    else:
        formatter = logging.Formatter(settings.LOGGING.LOG_FORMAT)
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # Create file handler
    file_handler = RotatingFileHandler(
        os.path.join(settings.LOGGING.LOG_DIR, "app.log"),
        maxBytes=10485760,  # 10MB
        backupCount=5
    )
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    
    return root_logger