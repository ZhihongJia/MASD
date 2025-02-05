# coding=utf-8
import sys
import json
IS_PY3 = sys.version_info.major == 3
if IS_PY3:
    from urllib.request import urlopen
    from urllib.request import Request
    from urllib.error import URLError
    from urllib.parse import urlencode
    from urllib.parse import quote_plus
else:
    import urllib2
    from urllib import quote_plus
    from urllib2 import urlopen
    from urllib2 import Request
    from urllib2 import URLError
    from urllib import urlencode

class DemoError(Exception):
    pass

class TTS():
    def __init__(self, PER=0):

        self.API_KEY = ''
        self.SECRET_KEY = ''
        self.PER = PER 
        self.SPD = 1
        self.PIT = 5
        self.VOL= 5
        self.AUE = 6

        FORMATS = {3: "mp3", 4: "pcm", 5: "pcm", 6: "wav"}
        self.FORMAT = FORMATS[self.AUE]

        self.CUID = "123456PYTHON"

        self.TTS_URL = 'http://tsn.baidu.com/text2audio'

        """  TOKEN start """

        self.TOKEN_URL = 'http://aip.baidubce.com/oauth/2.0/token'
        self.SCOPE = 'audio_tts_post'


    def fetch_token(self):
        print("fetch token begin")
        params = {'grant_type': 'client_credentials',
                'client_id': self.API_KEY,
                'client_secret': self.SECRET_KEY}
        post_data = urlencode(params)
        if (IS_PY3):
            post_data = post_data.encode('utf-8')
        req = Request(self.TOKEN_URL, post_data)
        try:
            f = urlopen(req, timeout=5)
            result_str = f.read()
        except URLError as err:
            print('token http response http code : ' + str(err.code))
            result_str = err.read()
        if (IS_PY3):
            result_str = result_str.decode()

        print(result_str)
        result = json.loads(result_str)
        print(result)
        if ('access_token' in result.keys() and 'scope' in result.keys()):
            if not self.SCOPE in result['scope'].split(' '):
                raise DemoError('scope is not correct')
            print('SUCCESS WITH TOKEN: %s ; EXPIRES IN SECONDS: %s' % (result['access_token'], result['expires_in']))
            return result['access_token']
        else:
            raise DemoError('MAYBE self.API_KEY or self.SECRET_KEY not correct: access_token or scope not found in token response')
    """  TOKEN end """

    def generate(self, text, save_path):
        token = self.fetch_token()
        tex = quote_plus(text)
        print(tex)
        params = {'tok': token, 'tex': tex, 'per': self.PER, 'spd': self.SPD, 'pit': self.PIT, 'vol': self.VOL, 'aue': self.AUE, 'cuid': self.CUID,
                'lan': 'zh', 'ctp': 1}

        data = urlencode(params)
        print('test on Web Browser' + self.TTS_URL + '?' + data)

        req = Request(self.TTS_URL, data.encode('utf-8'))
        has_error = False
        try:
            f = urlopen(req)
            result_str = f.read()

            headers = dict((name.lower(), value) for name, value in f.headers.items())

            has_error = ('content-type' not in headers.keys() or headers['content-type'].find('audio/') < 0)
        except  URLError as err:
            print('asr http response http code : ' + str(err.code))
            result_str = err.read()
            has_error = True

        save_file = "error.txt" if has_error else save_path
        with open(save_file, 'wb') as of:
            of.write(result_str)

        if has_error:
            if (IS_PY3):
                result_str = str(result_str, 'utf-8')
            print("tts api  error:" + result_str)

        print("result saved as :" + save_file)