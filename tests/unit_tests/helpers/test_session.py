import os
from unittest.mock import patch
import pytest
import requests
from pandasai.constants import DEFAULT_API_URL
from pandasai.exceptions import PandaAIApiCallError, PandaAIApiKeyError
from pandasai.helpers.session import Session, get_pandaai_session
from unittest.mock import patch, Mock
from pandasai.helpers.session import Session
from pandasai.exceptions import PandaAIApiCallError

@patch("pandasai.os.environ", {})
def test_session_init_without_api_key():
                                                                """Test that Session initialization raises PandaAIApiKeyError when no API key is provided"""
                                                                with pytest.raises(PandaAIApiKeyError) as exc_info:
                                                                                                                                Session()
                                                                assert (
                                                                                                                                str(exc_info.value)
                                                                                                                                == "PandaAI API key not found. Please set your API key using PandaAI.set_api_key() or by setting the PANDASAI_API_KEY environment variable."
                                                                )
@patch("pandasai.os.environ", {})
def test_session_init_with_none_api_key():
                                                                """Test that Session initialization raises PandaAIApiKeyError when API key is None"""
                                                                with pytest.raises(PandaAIApiKeyError) as exc_info:
                                                                                                                                Session(api_key=None)
                                                                assert (
                                                                                                                                str(exc_info.value)
                                                                                                                                == "PandaAI API key not found. Please set your API key using PandaAI.set_api_key() or by setting the PANDASAI_API_KEY environment variable."
                                                                )
@patch("pandasai.os.environ", {})
def test_session_init_with_api_key():
                                                                """Test that Session initialization works with a valid API key"""
                                                                session = Session(api_key="test-key")
                                                                assert session._api_key == "test-key"
@patch("pandasai.os.environ", {})
def test_session_init_with_default_api_url():
                                                                """Test that Session initialization uses DEFAULT_API_URL when no URL is provided"""
                                                                session = Session(api_key="test-key")
                                                                assert session._endpoint_url == DEFAULT_API_URL
@patch("pandasai.os.environ", {})
def test_session_init_with_custom_api_url():
                                                                """Test that Session initialization uses provided URL"""
                                                                custom_url = "https://custom.api.url"
                                                                session = Session(api_key="test-key", endpoint_url=custom_url)
                                                                assert session._endpoint_url == custom_url
@patch.dict(os.environ, {"PANDABI_API_KEY": "test-env-key"})
def test_session_init_with_env_api_key():
                                                                """Test that Session initialization works with API key from environment"""
                                                                session = Session()
                                                                assert session._api_key == "test-env-key"
@patch.dict(
                                                                os.environ,
                                                                {"PANDABI_API_KEY": "test-env-key", "PANDABI_API_URL": "https://env.api.url"},
)
def test_session_init_with_env_api_url():
                                                                """Test that Session initialization uses URL from environment"""
                                                                session = Session()
                                                                assert session._endpoint_url == "https://env.api.url"
@patch("pandasai.os.environ", {})
def test_get_pandaai_session_without_credentials():
                                                                """Test that get_pandaai_session raises PandaAIApiKeyError when no credentials are provided"""
                                                                with pytest.raises(PandaAIApiKeyError) as exc_info:
                                                                                                                                get_pandaai_session()
                                                                assert (
                                                                                                                                str(exc_info.value)
                                                                                                                                == "PandaAI API key not found. Please set your API key using PandaAI.set_api_key() or by setting the PANDASAI_API_KEY environment variable."
                                                                )
def test_get_pandaai_session_with_default_api_url():
                                                                """Test that get_pandaai_session uses DEFAULT_API_URL when no URL is provided"""
                                                                with patch.dict(os.environ, {"PANDABI_API_KEY": "test-key"}):
                                                                                                                                session = get_pandaai_session()
                                                                                                                                assert session._endpoint_url == DEFAULT_API_URL
@patch.dict(
                                                                os.environ,
                                                                {"PANDABI_API_KEY": "test-env-key", "PANDABI_API_URL": "http://test.url"},
)
def test_get_pandaai_session_with_env_credentials():
                                                                """Test that get_pandaai_session works with credentials from environment"""
                                                                session = get_pandaai_session()
                                                                assert isinstance(session, Session)
                                                                assert session._api_key == "test-env-key"
                                                                assert session._endpoint_url == "http://test.url"
@patch.dict(
                                                                os.environ,
                                                                {"PANDABI_API_KEY": "test-env-key", "PANDABI_API_URL": "https://env.api.url"},
)
def test_get_pandaai_session_with_env_api_url():
                                                                """Test that get_pandaai_session uses URL from environment"""
                                                                session = get_pandaai_session()
                                                                assert session._endpoint_url == "https://env.api.url"
@patch("requests.request")
def test_make_request_success(mock_request):
                                                                """Test successful API request"""
                                                                # Mock successful response
                                                                mock_response = mock_request.return_value
                                                                mock_response.status_code = 200
                                                                mock_response.json.return_value = {"data": "test_data"}
                                                                session = Session(api_key="test-key")
                                                                result = session.make_request("GET", "/test")
                                                                # Verify request was made correctly
                                                                mock_request.assert_called_once_with(
                                                                                                                                "GET",
                                                                                                                                DEFAULT_API_URL + "/api/test",
                                                                                                                                headers={
                                                                                                                                                                                                "x-authorization": "Bearer test-key",
                                                                                                                                                                                                "Content-Type": "application/json",
                                                                                                                                },
                                                                                                                                params=None,
                                                                                                                                data=None,
                                                                                                                                json=None,
                                                                                                                                timeout=300,
                                                                )
                                                                assert result == {"data": "test_data"}
@patch("requests.request")
def test_make_request_error_response(mock_request):
                                                                """Test API request with error response"""
                                                                # Mock error response
                                                                mock_response = mock_request.return_value
                                                                mock_response.status_code = 400
                                                                mock_response.json.return_value = {"message": "Bad request"}
                                                                session = Session(api_key="test-key")
                                                                with pytest.raises(PandaAIApiCallError) as exc_info:
                                                                                                                                session.make_request("POST", "/test")
                                                                assert str(exc_info.value) == "Bad request"
@patch("requests.request")
def test_make_request_network_error(mock_request):
                                                                """Test API request with network error"""
                                                                # Mock network error
                                                                mock_request.side_effect = requests.exceptions.RequestException("Network error")
                                                                session = Session(api_key="test-key")
                                                                with pytest.raises(PandaAIApiCallError) as exc_info:
                                                                                                                                session.make_request("GET", "/test")
                                                                assert "Request failed: Network error" in str(exc_info.value)
@patch("requests.request")
def test_make_request_custom_headers(mock_request):
                                                                """Test API request with custom headers"""
                                                                # Mock successful response
                                                                mock_response = mock_request.return_value
                                                                mock_response.status_code = 200
                                                                mock_response.json.return_value = {"data": "test_data"}
                                                                custom_headers = {"Custom-Header": "test-value"}
                                                                session = Session(api_key="test-key")
                                                                session.make_request("GET", "/test", headers=custom_headers)
                                                                # Verify custom headers were used
                                                                called_headers = mock_request.call_args[1]["headers"]
                                                                assert called_headers["Custom-Header"] == "test-value"
                                                                assert "x-authorization" not in called_headers
@patch("requests.request")
def test_make_request_success_non_json_response(mock_request):
                                                                """
                                                                Test successful API request with non-JSON response.
                                                                This test covers the scenario where the API returns a successful status code
                                                                but the response body is not JSON-formatted.
                                                                """
                                                                # Mock successful response with non-JSON content
                                                                mock_response = Mock()
                                                                mock_response.status_code = 200
                                                                mock_response.json.side_effect = ValueError("No JSON object could be decoded")
                                                                mock_response.content = b"Raw content"
                                                                mock_request.return_value = mock_response
                                                                session = Session(api_key="test-key")
                                                                result = session.make_request("GET", "/test")
                                                                # Verify request was made correctly
                                                                mock_request.assert_called_once_with(
                                                                                                                                "GET",
                                                                                                                                "https://api.pandabi.ai/api/test",  # Using DEFAULT_API_URL
                                                                                                                                headers={
                                                                                                                                                                                                "x-authorization": "Bearer test-key",
                                                                                                                                                                                                "Content-Type": "application/json",
                                                                                                                                },
                                                                                                                                params=None,
                                                                                                                                data=None,
                                                                                                                                json=None,
                                                                                                                                timeout=300,
                                                                )
                                                                # Verify that the raw response is returned
                                                                assert result == mock_response
@patch("requests.request")
def test_make_request_error_response_with_detail(mock_request):
                                """
                                Test API request with error response containing 'detail' instead of 'message'.
                                This test covers the scenario where the API returns an error status code
                                and the response body contains a 'detail' field for the error description.
                                """
                                # Mock error response
                                mock_response = mock_request.return_value
                                mock_response.status_code = 400
                                mock_response.json.return_value = {"detail": "Detailed error description"}
                                session = Session(api_key="test-key")
                                
                                with pytest.raises(PandaAIApiCallError) as exc_info:
                                                                session.make_request("POST", "/test")
                                assert str(exc_info.value) == "Detailed error description"
                                # Verify request was made correctly
                                mock_request.assert_called_once_with(
                                                                "POST",
                                                                "https://api.pandabi.ai/api/test",  # Using DEFAULT_API_URL
                                                                headers={
                                                                                                "x-authorization": "Bearer test-key",
                                                                                                "Content-Type": "application/json",
                                                                },
                                                                params=None,
                                                                data=None,
                                                                json=None,
                                                                timeout=300,
                                )
@patch("requests.request")
def test_make_request_success_201_status(mock_request):
                """
                Test successful API request with 201 status code.
                This test covers the scenario where the API returns a 201 status code,
                which is considered a success but not explicitly handled in the current implementation.
                """
                # Mock successful response with 201 status code
                mock_response = Mock()
                mock_response.status_code = 201
                mock_response.json.return_value = {"data": "created"}
                mock_request.return_value = mock_response
                session = Session(api_key="test-key")
                result = session.make_request("POST", "/create")
                # Verify request was made correctly
                mock_request.assert_called_once_with(
                                "POST",
                                "https://api.pandabi.ai/api/create",  # Using DEFAULT_API_URL
                                headers={
                                                "x-authorization": "Bearer test-key",
                                                "Content-Type": "application/json",
                                },
                                params=None,
                                data=None,
                                json=None,
                                timeout=300,
                )
                # Verify that the JSON response is returned
                assert result == {"data": "created"}
@pytest.mark.parametrize("http_method", ["get", "post", "patch", "put", "delete"])
def test_http_methods(http_method):
        """
        Test that each HTTP method (GET, POST, PATCH, PUT, DELETE) in the Session class
        correctly calls the make_request method with the appropriate HTTP method.
        This test ensures that the convenience methods for different HTTP requests
        are correctly implemented and delegate to the make_request method.
        """
        session = Session(api_key="test-key")
        
        with patch.object(Session, 'make_request') as mock_make_request:
                # Call the method dynamically
                method = getattr(session, http_method)
                method("/test-path", param1="value1")
                
                # Assert that make_request was called with the correct HTTP method
                mock_make_request.assert_called_once_with(
                        http_method.upper(),
                        "/test-path",
                        param1="value1"
                )