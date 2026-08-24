"""
Unit tests for the OpenAI LLM provider (FUTURE-RAG).

These tests isolate the OpenAILLM class using MagicMock and patching
to verify configuration, parameter propagation, API key handling,
and LlamaIndex LLM creation without making real API calls.
"""

from unittest.mock import patch

import pytest

from future_rag.llms.openai import OpenAILLM


# ---------------------------------------------------------------------------
# Unit Tests
# ---------------------------------------------------------------------------


class TestOpenAIInitialization:

    def test_init_default_values(self):
        """Verify that OpenAILLM uses the expected default values."""
        with patch.dict(
            "os.environ",
            {"OPENAI_API_KEY": "test-openai-key"},
        ):
            llm_provider = OpenAILLM()

        assert llm_provider.model == "gpt-4.1-mini"
        assert llm_provider.api_key == "test-openai-key"
        assert llm_provider.temperature == 0.2
        assert llm_provider.max_tokens == 4096

    def test_init_custom_values(self):
        """Verify that OpenAILLM respects custom configuration values."""
        llm_provider = OpenAILLM(
            model="custom-model",
            api_key="custom-key",
            temperature=0.5,
            max_tokens=2048,
        )

        assert llm_provider.model == "custom-model"
        assert llm_provider.api_key == "custom-key"
        assert llm_provider.temperature == 0.5
        assert llm_provider.max_tokens == 2048

    def test_api_key_from_environment(self):
        """Verify that the API key is loaded from OPENAI_API_KEY."""
        with patch.dict(
            "os.environ",
            {"OPENAI_API_KEY": "environment-key"},
        ):
            llm_provider = OpenAILLM()

        assert llm_provider.api_key == "environment-key"

    def test_explicit_api_key_takes_precedence(self):
        """Verify that an explicit API key takes precedence over the environment."""
        with patch.dict(
            "os.environ",
            {"OPENAI_API_KEY": "environment-key"},
        ):
            llm_provider = OpenAILLM(
                api_key="explicit-key"
            )

        assert llm_provider.api_key == "explicit-key"


class TestOpenAIGetLLM:

    @patch("future_rag.llms.openai.OpenAI")
    def test_get_llm(self, mock_openai):
        """Verify that get_llm() creates OpenAI with the correct parameters."""
        # Arrange
        provider = OpenAILLM(
            model="test-model",
            api_key="test-key",
            temperature=0.3,
            max_tokens=2048,
        )

        # Act
        llm = provider.get_llm()

        # Assert
        mock_openai.assert_called_once_with(
            model="test-model",
            api_key="test-key",
            temperature=0.3,
            max_tokens=2048,
        )

        assert llm == mock_openai.return_value