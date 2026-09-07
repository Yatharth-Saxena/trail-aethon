import pytest
from voice.command_parser.command_parser import command_parser, Intent

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
