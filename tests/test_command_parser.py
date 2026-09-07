import pytest
from voice.command_parser.command_parser import (
    command_parser,
    Intent,
    has_wake_word,
    strip_wake_word,
    split_wake_command,
)


def test_wake_word_stripping():
    """The wake phrase must be removed, leaving only the command text."""
    assert strip_wake_word("hey aethon what's the next step") == "what's the next step"
    assert strip_wake_word("Hey AETHON, what is the next step") == "what is the next step"
    assert strip_wake_word("aethon start the experiment") == "start the experiment"
    assert strip_wake_word("ok aethon pause the experiment") == "pause the experiment"

    # Misheard pronunciations strip just as well.
    assert strip_wake_word("hey than what am i doing") == "what am i doing"
    assert strip_wake_word("hetan what color is this") == "what color is this"
    assert strip_wake_word("ethane than take a snapshot") == "take a snapshot"

    # A bare wake call leaves nothing behind.
    assert strip_wake_word("hey aethon") == ""
    assert strip_wake_word("aethon") == ""
    assert strip_wake_word("hetan") == ""

    # Utterances with no wake word pass through untouched.
    assert strip_wake_word("what is the next step") == "what is the next step"
    assert strip_wake_word("") == ""


def test_wake_word_gating_contract():
    """split_wake_command mirrors the frontend gate: (had_wake, command)."""
    had_wake, command = split_wake_command("hey aethon what's the next step")
    assert had_wake is True
    assert command == "what's the next step"

    had_wake, command = split_wake_command("hey aethon")
    assert had_wake is True
    assert command == ""

    # Ambient chatter must not open a command session.
    had_wake, command = split_wake_command("Vikram chai pi li kya")
    assert had_wake is False
    assert command == "Vikram chai pi li kya"

    assert not has_wake_word("tell me the next step")
    assert has_wake_word("hey aethon tell me the next step")


def test_stripped_command_still_parses_to_intent():
    """Stripping the wake word must not change the recognised intent."""
    pairs = [
        ("hey aethon what's the next step", Intent.NEXT_STEP),
        ("hey aethon start the experiment", Intent.START_EXPERIMENT),
        ("aethon what am i doing", Intent.WHAT_AM_I_DOING),
        ("hetan what color is this", Intent.IDENTIFY_COLOR),
        ("ethane than take a snapshot", Intent.SNAPSHOT),
    ]
    for utterance, expected in pairs:
        assert command_parser.parse(utterance) == expected
        assert command_parser.parse(strip_wake_word(utterance)) == expected

def test_command_variations():
    assert command_parser.parse("Start the experiment") == Intent.START_EXPERIMENT
    assert command_parser.parse("Hey Aethon, start experiment") == Intent.START_EXPERIMENT
    assert command_parser.parse("Start") == Intent.START_EXPERIMENT

    assert command_parser.parse("What's the next step?") == Intent.NEXT_STEP
    assert command_parser.parse("What is the next step") == Intent.NEXT_STEP
    assert command_parser.parse("Tell me the next step") == Intent.NEXT_STEP

    assert command_parser.parse("Am I doing it right?") == Intent.CHECK_CURRENT_ACTION
    assert command_parser.parse("Am I doing this correctly?") == Intent.CHECK_CURRENT_ACTION

    assert command_parser.parse("Repeat the procedure") == Intent.REPEAT_STEP
    assert command_parser.parse("Repeat the step") == Intent.REPEAT_STEP
    assert command_parser.parse("Say that again") == Intent.REPEAT_STEP

    assert command_parser.parse("Pause the experiment") == Intent.PAUSE_EXPERIMENT
    assert command_parser.parse("Pause") == Intent.PAUSE_EXPERIMENT

    assert command_parser.parse("Resume the experiment") == Intent.RESUME_EXPERIMENT
    assert command_parser.parse("Continue") == Intent.RESUME_EXPERIMENT

    assert command_parser.parse("Reset the experiment") == Intent.RESET_EXPERIMENT
    assert command_parser.parse("Reset") == Intent.RESET_EXPERIMENT

    assert command_parser.parse("Stop the experiment") == Intent.STOP_EXPERIMENT
    assert command_parser.parse("Stop") == Intent.STOP_EXPERIMENT

def test_greetings_and_conversation():
    assert command_parser.parse("Hello") == Intent.GREETING
    assert command_parser.parse("Hey Aethon") == Intent.GREETING
    assert command_parser.parse("Who are you") == Intent.GREETING
    assert command_parser.parse("What is your name") == Intent.GREETING
    assert command_parser.parse("How are you") == Intent.GREETING
    assert command_parser.parse("Thank you") == Intent.GREETING
    assert command_parser.parse("Can you hear me") == Intent.GREETING
    assert command_parser.parse("Goodbye") == Intent.GREETING
    assert command_parser.parse("Namaste") == Intent.GREETING

def test_unrelated_queries_and_service():
    from voice.command_parser.aethon_command_service import aethon_command_service

    # Unrelated query from web dashboard returns the polite fixed response
    res1 = aethon_command_service.handle_command("Who won the football match?", source="web")
    assert "focused on your experiment mission" in res1["response"]

    # Conversational greeting returns friendly response
    res2 = aethon_command_service.handle_command("Who are you?", source="web")
    assert "AETHON" in res2["response"]

    res3 = aethon_command_service.handle_command("How are you?", source="web")
    assert "nominal" in res3["response"]

    # Hardware mic with wake word + unrelated query returns fixed response
    res4 = aethon_command_service.handle_command("Hey Aethon, tell me a joke", source="hardware_mic")
    assert "focused on your experiment mission" in res4["response"]

    # Hardware mic with ambient chatter without wake word is ignored
    res5 = aethon_command_service.handle_command("Vikram chai pi li kya", source="hardware_mic")
    assert res5["intent"] == "IGNORE"
    assert res5["response"] == ""

def test_misheard_wake_word_pronunciations():
    from voice.command_parser.aethon_command_service import aethon_command_service
    from voice.command_parser.command_parser import has_wake_word

    # Check wake word detection on common speech-to-text mishearings
    assert has_wake_word("hey than")
    assert has_wake_word("hetan")
    assert has_wake_word("Ethane than")
    assert has_wake_word("hey Ethan")
    assert has_wake_word("Ethan what am I doing")
    assert has_wake_word("hey than what am i doing")
    assert not has_wake_word("Tumhari Batti Nahin chal rahe Iske")
    assert not has_wake_word("abhi tak To Yahi bol raha tha")

    # Check parsing
    assert command_parser.parse("hey than") == Intent.GREETING
    assert command_parser.parse("hetan") == Intent.GREETING
    assert command_parser.parse("Ethane than") == Intent.GREETING
    assert command_parser.parse("hey Ethan") == Intent.GREETING
    assert command_parser.parse("Ethan what am I doing") == Intent.WHAT_AM_I_DOING
    assert command_parser.parse("hey than what am i doing") == Intent.WHAT_AM_I_DOING

    # Check normalization: Chat bubble displays correct "AETHON"
    norm1 = aethon_command_service._normalize_aethon_spelling("hey than")
    assert "Hey AETHON" in norm1
    assert "than" not in norm1.lower()

    norm2 = aethon_command_service._normalize_aethon_spelling("hetan")
    assert "Hey AETHON" in norm2

    norm3 = aethon_command_service._normalize_aethon_spelling("Ethane than")
    assert "Hey AETHON" in norm3

    norm4 = aethon_command_service._normalize_aethon_spelling("hey Ethan")
    assert "Hey AETHON" in norm4

    norm5 = aethon_command_service._normalize_aethon_spelling("Ethan what am I doing")
    assert "AETHON" in norm5
    assert "Ethan" not in norm5

    # Test handling via service with hardware_mic
    res = aethon_command_service.handle_command("hey than", source="hardware_mic")
    assert res["intent"] == "GREETING"
    assert "Commander" in res["response"]

    # Test 'hey thanks', 'hey then', 'aethan'
    assert has_wake_word("hey thanks")
    assert has_wake_word("hey then")
    assert has_wake_word("aethan")
    res_thanks = aethon_command_service.handle_command("hey thanks", source="hardware_mic")
    assert res_thanks["intent"] == "GREETING"
    assert "Commander" in res_thanks["response"]
