
1. in the voice/client.py the stt, tts is abstract but there would be no way that most of providers' stt, and tts provides those methods. uuh looking at the registry.py now it makes sense.


2. the stt, tts in the registry.py shouldn't follow the abstract stt, tts classes in the voice/client.py.

3. why the voice_ws_adapter does not inherit from the ws_adapter that it directly inherits from ProtocolAdapter. probably need to discuss about some kind of mixin pattern.

4. the voice/bridge/pipecat.py  says pipecate service is stream or frame based while the thin implementation is batch. need to understand what is the difference.
this was the note mentioned there:
NOTE (scaffold): Pipecat services are streaming / frame-based, whereas the thin
channel calls them in batch. The batch<->frame glue is intentionally left as a
focused follow-up; construction lazily imports Pipecat and fails with an
actionable message if the extra is missing. See ``docs/voice-plan.md``.



5. in adapters/voice_ws_adapter.py if turn does not have a response then it is automatically counted as agent failure. but at the end maybe the agent doesn't need to response because at the end of the previous turn the agent said "bye" and in the begining of the first turn the testing agent replies with "bye" and the agent doesn't respond and just ends the call or chat or whatever.
then again in the same file it also sets unintelligible error, it is not clear how it can detect that there was a voice but it is not understandable.

6. same issue of getting blank response as agentFailure.no_response is applied agents/user_simulator.py as well.