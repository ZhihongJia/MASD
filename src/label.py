class Label_Ref():
    def __init__(self) -> None:
        '''
        声母:0 双唇音: b p m 
            1 齿唇音: f
            2 舌尖前音: z c s
            3 舌尖中音; d t n l
            4 舌尖后音: zh ch sh r
            5 舌面音: j q x
            6 舌根音: g k h
            7 无声母
        韵母：开口呼: 韵母开头不是i、u、v的韵母属于开口呼 0
             齐齿呼: 韵母开头是i的韵母属于齐齿呼 1
             合口呼: 韵母开头是u的韵母属于合口呼 2
             撮口呼: 韵母开头是v的韵母属于撮口呼 3
        '''
        self.label = {'飘':{'word': 0, 'initial': 'p', 'initial_class': 1, 'initial_class_8':0, 'final': 'iao', 'final_class':1, 'tone_class':0},
                      '方':{'word': 1, 'initial': 'f', 'initial_class': 3, 'initial_class_8':1, 'final': 'ang', 'final_class':0, 'tone_class':0},
                      '丢':{'word': 2, 'initial': 'd', 'initial_class': 4, 'initial_class_8':3, 'final':  'iu', 'final_class':1, 'tone_class':0},
                      '他':{'word': 3, 'initial': 't', 'initial_class': 5, 'initial_class_8':3, 'final':   'a', 'final_class':0, 'tone_class':0},
                      '天':{'word': 4, 'initial': 't', 'initial_class': 5, 'initial_class_8':3, 'final': 'ian', 'final_class':1, 'tone_class':0},
                      '光':{'word': 5, 'initial': 'g', 'initial_class': 8, 'initial_class_8':6, 'final':'uang', 'final_class':2, 'tone_class':0},
                      '均':{'word': 6, 'initial': 'j', 'initial_class':11, 'initial_class_8':5, 'final':  'vn', 'final_class':3, 'tone_class':0},
                      '缺':{'word': 7, 'initial': 'q', 'initial_class':12, 'initial_class_8':5, 'final':  've', 'final_class':3, 'tone_class':0},
                      '出':{'word': 8, 'initial':'ch', 'initial_class':15, 'initial_class_8':4, 'final':   'u', 'final_class':2, 'tone_class':0},
                      '说':{'word': 9, 'initial':'sh', 'initial_class':16, 'initial_class_8':4, 'final':  'uo', 'final_class':2, 'tone_class':0},
                      '扔':{'word':10, 'initial': 'r', 'initial_class':17, 'initial_class_8':4, 'final': 'eng', 'final_class':0, 'tone_class':0},
                      '三':{'word':11, 'initial': 's', 'initial_class':20, 'initial_class_8':2, 'final':  'an', 'final_class':0, 'tone_class':0},
                      '没':{'word':12, 'initial': 'm', 'initial_class': 2, 'initial_class_8':0, 'final':  'ei', 'final_class':0, 'tone_class':1},
                      '明':{'word':13, 'initial': 'm', 'initial_class': 2, 'initial_class_8':0, 'final': 'ing', 'final_class':1, 'tone_class':1},
                      '年':{'word':14, 'initial': 'n', 'initial_class': 6, 'initial_class_8':3, 'final': 'ian', 'final_class':1, 'tone_class':1},
                      '结':{'word':15, 'initial': 'j', 'initial_class':11, 'initial_class_8':5, 'final':  'ie', 'final_class':1, 'tone_class':1},
                      '穷':{'word':16, 'initial': 'q', 'initial_class':12, 'initial_class_8':5, 'final':'iong', 'final_class':1, 'tone_class':1},
                      '床':{'word':17, 'initial':'ch', 'initial_class':15, 'initial_class_8':4, 'final':'uang', 'final_class':2, 'tone_class':1},
                      '人':{'word':18, 'initial': 'r', 'initial_class':17, 'initial_class_8':4, 'final':  'en', 'final_class':0, 'tone_class':1},
                      '昨':{'word':19, 'initial': 'z', 'initial_class':18, 'initial_class_8':2, 'final':  'uo', 'final_class':2, 'tone_class':1},
                      '从':{'word':20, 'initial': 'c', 'initial_class':19, 'initial_class_8':2, 'final': 'ong', 'final_class':0, 'tone_class':1},
                      '才':{'word':21, 'initial': 'c', 'initial_class':19, 'initial_class_8':2, 'final':  'ai', 'final_class':0, 'tone_class':1},
                      '随':{'word':22, 'initial': 's', 'initial_class':20, 'initial_class_8':2, 'final':  'ui', 'final_class':2, 'tone_class':1},
                      '而':{'word':23, 'initial':'er', 'initial_class':23, 'initial_class_8':7, 'final':  'er', 'final_class':0, 'tone_class':1},
                      '把':{'word':24, 'initial': 'b', 'initial_class': 0, 'initial_class_8':0, 'final':   'a', 'final_class':0, 'tone_class':2},
                      '品':{'word':25, 'initial': 'p', 'initial_class': 1, 'initial_class_8':0, 'final':  'in', 'final_class':1, 'tone_class':2},
                      '你':{'word':26, 'initial': 'n', 'initial_class': 6, 'initial_class_8':3, 'final':   'i', 'final_class':1, 'tone_class':2},
                      '旅':{'word':27, 'initial': 'l', 'initial_class': 7, 'initial_class_8':3, 'final':   'v', 'final_class':3, 'tone_class':2},
                      '两':{'word':28, 'initial': 'l', 'initial_class': 7, 'initial_class_8':3, 'final':'iang', 'final_class':1, 'tone_class':2},
                      '缓':{'word':29, 'initial': 'h', 'initial_class':10, 'initial_class_8':6, 'final': 'uan', 'final_class':2, 'tone_class':2},
                      '选':{'word':30, 'initial': 'x', 'initial_class':13, 'initial_class_8':5, 'final': 'van', 'final_class':3, 'tone_class':2},
                      '准':{'word':31, 'initial':'zh', 'initial_class':14, 'initial_class_8':4, 'final':  'un', 'final_class':2, 'tone_class':2},
                      '早':{'word':32, 'initial': 'z', 'initial_class':18, 'initial_class_8':2, 'final':  'ao', 'final_class':0, 'tone_class':2},
                      '有':{'word':33, 'initial': 'y', 'initial_class':21, 'initial_class_8':7, 'final':  'ou', 'final_class':1, 'tone_class':2},
                      '我':{'word':34, 'initial': 'w', 'initial_class':22, 'initial_class_8':7, 'final':   'o', 'final_class':2, 'tone_class':2},
                      '耳':{'word':35, 'initial':'er', 'initial_class':23, 'initial_class_8':7, 'final':  'er', 'final_class':0, 'tone_class':2},
                      '不':{'word':36, 'initial': 'b', 'initial_class': 0, 'initial_class_8':0, 'final':   'u', 'final_class':2, 'tone_class':3},
                      '费':{'word':37, 'initial': 'f', 'initial_class': 3, 'initial_class_8':1, 'final':  'ei', 'final_class':0, 'tone_class':3},
                      '第':{'word':38, 'initial': 'd', 'initial_class': 4, 'initial_class_8':3, 'final':   'i', 'final_class':1, 'tone_class':3},
                      '个':{'word':39, 'initial': 'g', 'initial_class': 8, 'initial_class_8':6, 'final':   'e', 'final_class':0, 'tone_class':3},
                      '看':{'word':40, 'initial': 'k', 'initial_class': 9, 'initial_class_8':6, 'final':  'an', 'final_class':0, 'tone_class':3},
                      '快':{'word':41, 'initial': 'k', 'initial_class': 9, 'initial_class_8':6, 'final': 'uai', 'final_class':2, 'tone_class':3},
                      '话':{'word':42, 'initial': 'h', 'initial_class':10, 'initial_class_8':6, 'final':  'ua', 'final_class':2, 'tone_class':3},
                      '下':{'word':43, 'initial': 'x', 'initial_class':13, 'initial_class_8':5, 'final':  'ia', 'final_class':1, 'tone_class':3},
                      '这':{'word':44, 'initial':'zh', 'initial_class':14, 'initial_class_8':4, 'final':   'e', 'final_class':0, 'tone_class':3},
                      '上':{'word':45, 'initial':'sh', 'initial_class':16, 'initial_class_8':4, 'final': 'ang', 'final_class':0, 'tone_class':3},
                      '要':{'word':46, 'initial': 'y', 'initial_class':21, 'initial_class_8':7, 'final':  'ao', 'final_class':1, 'tone_class':3},
                      '问':{'word':47, 'initial': 'w', 'initial_class':22, 'initial_class_8':7, 'final':  'en', 'final_class':2, 'tone_class':3}}