from langchain_core.tools import tool

from api_tool_agent.api.client import predict_titanic_fare


@tool
def titanic_tool(
    pclass: int,
    sex: str,
    age: float,
    embarked: str,
    family_size: int,
) -> dict:
    """
    Predict the estimated Titanic passenger fare.

    Use this tool when the user asks for a Titanic fare prediction
    based on passenger information.

    Args:
        pclass:
            Passenger class.
            Valid values: 1, 2, or 3.

        sex:
            Passenger sex.
            Valid values: "male" or "female".

        age:
            Passenger age in years.

        embarked:
            Embarkation port.
            S = Southampton
            C = Cherbourg
            Q = Queenstown

        family_size:
            Total family size including the passenger.
    """

    return predict_titanic_fare(
        pclass=pclass,
        sex=sex,
        age=age,
        embarked=embarked,
        family_size=family_size,
    )
