import pytest
from datetime import datetime, UTC
from backend.models.transcript import Transcript, Turn

def test_add_trun_updates_list():
    transcript = Transcript(scenario_id="sc_1", session_id="sess_1", protocol="http")
    assert len(transcript.turns) == 0

    turn = Turn(user_message="Hello", agent_response="Hi!")
    transcript.add_turn(turn)

    assert len(transcript.turns)== 1
    assert transcript.turns[0].user_message == "Hello"

def test_transcript_finalize_sets_ended_at():
    transcript = Transcript(scenario_id="sc_1", session_id="sess_1", protocol="http")
    assert transcript.ended_at is None

    transcript.finalize()
    assert isinstance(transcript.ended_at, datetime)


def test_has_error_property():
    transcript = Transcript(scenario_id="sc_1", session_id="sess_1", protocol="http")
    transcript.add_turn(Turn(user_message="Hello", agent_response="Hi"))
    assert transcript.has_error is False

    transcript.add_turn(Turn(user_message="Bye", agent_response="", error="Timeout error"))
    assert transcript.has_error is True

def test_last_error_retrieval():
    transcript = Transcript(scenario_id="sc_1", session_id="sess_1", protocol="http")
    assert transcript.last_error is None

    transcript.add_turn(Turn(user_message="Hello", agent_response="", error="Error 1"))
    transcript.add_turn(Turn(user_message="Retry", agent_response="Success"))

    assert transcript.last_error == "Error 1"