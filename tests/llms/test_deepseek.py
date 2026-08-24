"""
Unit tests for the DeepSeek LLM provider (FUTURE-RAG).

These tests isolate the DeepSeekLLM class using MagicMock and patching
to verify configuration, parameter propagation, API key handling,
and LlamaIndex LLM creation without making real API calls.
"""

from unittest.mock import MagicMock, patch

import pytest

from future_rag.llms.deepseek import DeepSeekLLM


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_openai_like():
    """Provides a mock LlamaIndex OpenAILike instance."""
    return MagicMock(name="mock_openai_like")


# ---------------------------------------------------------------------------
# Unit Tests
# ---------------------------------------------------------------------------


class TestDeepSeekInitialization:

    def test_init_default_values(self):
        """Verify that DeepSeekLLM uses the expected default values."""
        with patch.dict(
            "os.environ",
            {"DEEPSEEK_API_KEY": "test-deepseek-key"},
        ):
            llm_provider = DeepSeekLLM()

        assert llm_provider.model == "deepseek-v4-flash"
        assert llm_provider.api_key == "test-deepseek-key"
        assert llm_provider.api_base == "https://api.deepseek.com"
        assert llm_provider.temperature == 0.2
        assert llm_provider.context_window == 64000
        assert llm_provider.max_tokens == 4096

    def test_init_custom_values(self):
        """Verify that DeepSeekLLM respects custom configuration values."""
        llm_provider = DeepSeekLLM(
            model="custom-model",
            api_key="custom-key",
            api_base="https://custom-endpoint.com",
            temperature=0.5,
            context_window=32000,
            max_tokens=2048,
        )

        assert llm_provider.model == "custom-model"
        assert llm_provider.api_key == "custom-key"
        assert llm_provider.api_base == "https://custom-endpoint.com"
        assert llm_provider.temperature == 0.5
        assert llm_provider.context_window == 32000
        assert llm_provider.max_tokens == 2048

    def test_api_key_from_environment(self):
        """Verify that the API key is loaded from DEEPSEEK_API_KEY."""
        with patch.dict(
            "os.environ",
            {"DEEPSEEK_API_KEY": "environment-key"},
        ):
            llm_provider = DeepSeekLLM()

        assert llm_provider.api_key == "environment-key"

    def test_explicit_api_key_takes_precedence(self):
        """Verify that an explicit API key takes precedence over the environment."""
        with patch.dict(
            "os.environ",
            {"DEEPSEEK_API_KEY": "environment-key"},
        ):
            llm_provider = DeepSeekLLM(
                api_key="explicit-key"
            )

        assert llm_provider.api_key == "explicit-key"


class TestDeepSeekGetLLM:

    @patch("future_rag.llms.deepseek.OpenAILike")
    def test_get_llm(self, mock_openai_like):
        """Verify that get_llm() creates OpenAILike with the correct parameters."""
        # Arrange
        provider = DeepSeekLLM(
            model="test-model",
            api_key="test-key",
            api_base="https://test-api.com",
            temperature=0.3,
            context_window=32000,
            max_tokens=2048,
        )

        # Act
        llm = provider.get_llm()

        # Assert
        mock_openai_like.assert_called_once_with(
            model="test-model",
            api_key="test-key",
            api_base="https://test-api.com",
            temperature=0.3,
            context_window=32000,
            max_tokens=2048,
            is_chat_model=True,
        )

        assert llm == mock_openai_like.return_value