# Copyright 2023 Sony Group Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from wsgiref import simple_server
import argparse
import falcon
from interactive_tts import InteractiveTTS
import soundfile as sf
import io
import json

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='libritts_tvcgmm_k5', help='Model checkpoint name')
    parser.add_argument('--vocoder', type=str, default=None, help='Vocoder checkpoint name')
    parser.add_argument('--step', type=int, default=40000, help='Model step to load')
    parser.add_argument('--device', type=str, default='cpu', help='Device to run on', choices=['cpu', 'cuda'])
    parser.add_argument('--port', type=int, default=9000)
    
    args = parser.parse_args()
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    print('Loading model checkpoint...')
    synthesizer = InteractiveTTS(args.checkpoint, args.vocoder, args.step, device=args.device)

    STORE = {'image': None}

    class UIResource:
        def on_get(self, req, res):
            res.content_type = 'text/html'
            res.body = open('demo.html').read().replace('<!INJECT N_SPEAKERS>', str(synthesizer.preprocess_config["stats"]["n_speakers"])).replace('<!INJECT ENABLE_TVCGMM>', 'true' if synthesizer.model_config["tvcgmm"]["enabled"] else 'false')

    class SynthesisResource:
        def on_get(self, req, res):
            if not req.params.get('text'):
                raise falcon.HTTPBadRequest()
            wav, _, img, *_ = synthesizer.synthesize(
                req.params.get('text'), 
                duration=float(req.params.get('velocity')), 
                pitch=float(req.params.get('pitch')), 
                energy=float(req.params.get('energy')), 
                speaker_id=int(req.params.get('speaker')), 
                conditional=req.params.get('sampling')=='conditional'
            )
            STORE['image'] = img
            io_out = io.BytesIO()
            sf.write(io_out, wav, 22050, format='wav')
            res.data = io_out.getvalue()
            res.content_type = 'audio/wav'

    class ImageResource:
        def on_get(self, req, res):
            if STORE['image'] is None:
                raise falcon.HTTPBadRequest()
            io_out = io.BytesIO()
            STORE['image'].savefig(io_out, format = 'png')
            res.data = io_out.getvalue()
            res.content_type = 'image/png'


    api = falcon.API()
    api.add_route('/synthesize', SynthesisResource())
    api.add_route('/image', ImageResource())
    api.add_route('/', UIResource())

    print('Serving on port %d' % args.port)
    simple_server.make_server('0.0.0.0', args.port, api).serve_forever()
