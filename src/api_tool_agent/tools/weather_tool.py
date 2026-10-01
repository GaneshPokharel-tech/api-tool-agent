from langchain_core.tools import tool

from api_tool_agent.api.client import get_weather


@tool
def weather_tool(
    city: str,
    country_code: str | None = None,
) -> dict:
    """
    Get the current weather for a city.

    Use this tool when the user asks about current weather,
    temperature, humidity, wind speed, or weather conditions
    for a location.

    Args:
        city:
            City or municipality name.

        country_code:
            Optional two-letter ISO country code.

            Examples:
            NP = Nepal
            US = United States
            AU = Australia
    """

    return get_weather(
        city=city,
        country_code=country_code,
    )
