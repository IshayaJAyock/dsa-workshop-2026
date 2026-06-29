"""Package notebook outputs for the instant multimodal Gradio preview."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.output_formatting import media_path_for_gradio


@dataclass
class TourSession:
    """Everything the tour guide UI needs — built once in the notebook."""

    base_city: str
    base_country: str
    destination_city: str
    destination_country: str
    base_currency: str
    destination_currency: str
    traveller_profile: str
    brief_md: str
    tool_df: pd.DataFrame
    places_df: pd.DataFrame
    poster_path: str

    @property
    def route_label(self) -> str:
        return (
            f"{self.base_city}, {self.base_country} → "
            f"{self.destination_city}, {self.destination_country}"
        )


def pack_tour_session(
    base_city: str,
    base_country: str,
    destination_city: str,
    destination_country: str,
    base_currency: str,
    destination_currency: str,
    traveller_profile: str,
    brief_md: str,
    tool_df: pd.DataFrame,
    places_df: pd.DataFrame,
    poster_path: str | Any,
) -> TourSession:
    """Collect notebook variables into one object for the Gradio preview app."""
    return TourSession(
        base_city=base_city,
        base_country=base_country,
        destination_city=destination_city,
        destination_country=destination_country,
        base_currency=base_currency,
        destination_currency=destination_currency,
        traveller_profile=traveller_profile,
        brief_md=brief_md,
        tool_df=tool_df,
        places_df=places_df,
        poster_path=media_path_for_gradio(poster_path) or "",
    )
