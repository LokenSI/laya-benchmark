"""Use the publisher's verified Decision 2.0 runtime without changing inference math."""
from .alternatives_adapters import typed_questions, decode_typed, answer_probabilities
from .decision2_prepare import native_package

class Decision2:
    def __init__(self, meta):
        import importlib.metadata
        native = native_package(meta['path'], meta['revision'])
        self.model = native.Decision2.from_pretrained(meta['path'], device='cuda:0', threads=8,
                                                     graphs=False, kernels=False, share_context=False)
        backend = self.model.backend
        self.metadata = {'precision': 'Publisher BF16-exact linear residency; FP32 embeddings, norms and decision head; BF16 autocast',
                         'interface': 'Verified native Decision2.system_one; exact path, eager CUDA',
                         'max_tokens': self.model.max_input_tokens, 'option_limit': 255,
                         'context_policy': 'Native overflow rejection, no truncation',
                         'temperatures': backend.temperatures, 'residency': backend.residency,
                         'runtime_note': 'Windows NVIDIA with installed Transformers/FLA attention; no ROCm kernel or latency parity claim',
                         'graphs': False, 'kernels': False, 'share_context': False}
        self.metadata['portability_shim'] = 'Canonical POSIX fingerprint path names on Windows; exact publisher file roster and scored digest still checked; no inference changes'
        self.metadata['runtime_versions'] = {package: importlib.metadata.version(package) for package in
                                            ['torch', 'transformers', 'flash-linear-attention', 'triton-windows']}

    def batch(self, rows):
        results = []
        for row in rows:
            questions, mapping = typed_questions(row['questions'])
            response = self.model.system_one(state=row['state'], questions=questions)
            errors = {key: answer['error'] for key, answer in response['answers'].items()
                      if isinstance(answer, dict) and answer.get('error')}
            if errors:
                if all(reason in ('max_length_exceeded', 'invalid_question') for reason in errors.values()):
                    raise ValueError('Decision 2.0 native rejection: ' + str(errors))
                raise RuntimeError('Decision 2.0 invalid output: ' + str(errors))
            result = decode_typed(answer_probabilities(response['answers']), row['questions'], mapping)
            for key, question in row['questions'].items():
                if question['type'] == 'choice':
                    result['pred'][key] = [response['answers'][key]['choice']]
                elif question['type'] == 'noul':
                    result['pred'][key] = ['true' if result['probabilities'][key]['true'] >= .5 else 'false']
            result['native_answers'] = response['answers']
            result['native_usage'] = response.get('usage')
            results.append(result)
        return results
