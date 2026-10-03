import asyncio
import json
import pytest
from laya_bench import decision2_stress as stress

def test_probability_drift_does_not_hide_option_roster_changes():
    reference = {'pred': {'q': ['a']}, 'probabilities': {'q': {'a': .6, 'b': .4}}}
    result = {'pred': {'q': ['b']}, 'probabilities': {'q': {'a': .45, 'b': .55}}}
    assert stress.compare(reference, result) == {'labels_changed': True, 'max_probability_delta': pytest.approx(.15)}
    with pytest.raises(ValueError, match='option roster'):
        stress.compare(reference, {'pred': {'q': ['a']}, 'probabilities': {'q': {'a': 1}}})
    assert stress.compare(reference, {'error': 'timeout'})['labels_changed'] is None

def test_real_http_client_enforces_concurrency_and_sends_no_labels(tmp_path, monkeypatch):
    from aiohttp import web
    monkeypatch.setattr(stress, 'OUT', tmp_path)
    monkeypatch.setattr(stress, 'LEVELS', [1, 8])
    stress.prepare()
    async def exercise():
        inflight = peak = 0
        async def respond(request):
            nonlocal inflight, peak
            payload = await request.json()
            assert set(payload) == {'model', 'state', 'questions'}
            inflight += 1
            peak = max(peak, inflight)
            await asyncio.sleep(.001)
            answers = {}
            predictions = {}
            for key, question in payload['questions'].items():
                options = list(question.get('criteria') or ['false', 'true'])
                answers[key] = {option: .6 if i == 0 else .4/(len(options)-1) for i, option in enumerate(options)}
                predictions[key] = [options[0]]
            inflight -= 1
            return web.json_response({'pred': predictions, 'probabilities': answers,
                                      'service_timing': {'queue_seconds': 0, 'compute_seconds': .001}})
        app = web.Application()
        app.router.add_post('/v1/systemone', respond)
        async def status(request):
            return web.json_response({'queued_requests': 0})
        app.router.add_get('/status', status)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, '127.0.0.1', 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        try:
            await stress.run('fake-model', port)
        finally:
            await runner.cleanup()
        assert peak == 8
    asyncio.run(exercise())
    result = json.loads((tmp_path/'fake-model/summary.json').read_text())
    assert result['status'] == 'complete'
    assert [entry['requests'] for entry in result['concurrency']] == [240, 240]
    assert all(entry['errors'] == {} and entry['label_flips'] == 0 for entry in result['concurrency'])
    assert all(entry['max_probability_delta'] == 0 for entry in result['concurrency'])
    assert result['question_capacity'][-1]['questions'] == 512
    saved = [json.loads(line) for line in (tmp_path/'fake-model/requests.jsonl').read_text().splitlines()]
    assert len(saved) == 480
    capacity = [json.loads(line) for line in (tmp_path/'fake-model/question-capacity.jsonl').read_text().splitlines()]
    assert len(capacity) == 18
    assert len(capacity[-1]['answer']['pred']) == 512
