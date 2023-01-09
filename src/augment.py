import logging
import random
import jieba
import re
import json
jieba.setLogLevel(logging.INFO)
for i in range(100):
    jieba.add_word(f"FORMULA{i}")
# https://github.com/chatopera/Synonyms
# import synonyms


class Augment():
    def __init__(self, args, p=0.8):
        self.text_func_list = [
            self.random_swap_word,
            self.random_delete_word,
            # self.random_insert_word,
            # self.synonym_replace,
        ]
        self.formula_func_list = [
            self.variable_rename,
            self.variable_replace,
            self.op_synonym_replace,
            self.num_replace,
        ]
        self.ques_func_list = [
            self.random_shuffle_clause,
            self.random_clause_insert
        ]
        self.p = p
        self.args = args

        self.zh_pattern = re.compile(u'[\u4e00-\u9fa5\。\，\（\）\、\：\；\‘\’\Ⅰ\Ⅱ\Ⅲ\“\”]+')
        self.pure_zh_pattern = re.compile(u'[\u4e00-\u9fa5]+')
        # variable
        self.x_variable = ['x', 'y', 'z', 'u', 'v', 'w']
        self.greek_variable = ['\\alpha', '\\beta', '\\theta', '\\delta', '\\epsilon', '\\lambda', '\\mu', '\\omega', '\\rho', '\\phi', '\\sigma']
        self.big_variable = ['A', 'B', "C", 'D', 'E', 'F']

        self.vr_op = ['+', '-', '*', '/', '^', '\\frac']
        self.synonym_dict = {}

        # operator
        with open('./op_snynonym.json', 'r') as f:
            op_synonym = json.load(f)
        self.op_synonym = {}
        for l in op_synonym:
            for _op in l:
                self.op_synonym[_op] = l
                self.op_synonym[_op].remove(_op)

        self.sub_ques = ['(I)', '(1)', '（1）', '（I）', '(Ⅰ)', '（Ⅰ）']

    # ------- augmentation for text ------------
    def random_swap_word(self, text):
        if random.random() > self.p:
            return text
        else:
            words = jieba.lcut(text)
            if len(words) <= 1:
                return text
            w_indexes = []
            for i, w in enumerate(words):
                if not w.startswith('FORMULA'):
                    w_indexes.append(i)
            if len(w_indexes) < 2:
                return text
            index1 = random.choice(w_indexes)
            index2 = random.choice(w_indexes)
            while index2 == index1:
                index2 = random.choice(w_indexes)
            words[index1], words[index2] = words[index2], words[index1]
        return "".join(words)

    def random_delete_word(self, text):
        if random.random() > self.p:
            return text
        else:
            text = jieba.lcut(text)
            if len(text) > 1:
                delete_index = random.randint(0, len(text)-1)
                del text[delete_index]
        return "".join(text)

    def synonym_replace(self, text):
        if random.random() > self.p:
            return text
        else:
            words = jieba.lcut(text)
            cnt = 0
            while cnt < 2 * len(words):
                cnt += 1
                seleted_word = words[random.randint(0, len(words)-1)]
                if seleted_word[:7] == 'FORMULA':
                    continue
                if seleted_word in self.synonym_dict:
                    synonyms_list = self.synonym_dict[seleted_word]
                else:
                    synonyms_list = synonyms.nearby(seleted_word, 4)[0]
                    self.synonym_dict[seleted_word] = synonyms_list
                if len(synonyms_list) > 1:
                    new_word = random.choice(synonyms_list[1:])
                    words =[new_word if w == seleted_word else w for w in words]
                    break
        return "".join(words)

    def random_insert_word(self, text):
        if random.random() > self.p:
            return text
        else:
            words = jieba.lcut(text)
            cnt = 0
            while cnt < 2*len(words):
                cnt += 1
                seleted_word = words[random.randint(0, len(words)-1)]
                if seleted_word[:7] == 'FORMULA':
                    continue
                if seleted_word in self.synonym_dict:
                    synonyms_list = self.synonym_dict[seleted_word]
                else:
                    synonyms_list = synonyms.nearby(seleted_word, 4)[0]
                    self.synonym_dict[seleted_word] = synonyms_list
                if len(synonyms_list) > 1:
                    new_word = random.choice(synonyms_list[1:])
                    random_idx = random.randint(0, len(words)-1)
                    words.insert(random_idx, new_word)
                    break
        return "".join(words)

    def save_synonym_dict(self, path):
        print(f"[Augment] # of synonym_dict: {len(self.synonym_dict)}")
        with open(path, 'w') as f:
            json.dump(self.synonym_dict, f, ensure_ascii=False, indent=2)

    def load_synonym_dict(self, path):
        with open(path, 'r') as f:
            self.synonym_dict = json.load(f)
        print(f"[Augment] # of synonym_dict: {len(self.synonym_dict)}")

    # ------- augmentation for fomula ------------
    def variable_rename(self, l_formula):
        if random.random() > self.p:
            return l_formula
        else:
            l = [
                self.x_variable,
                self.greek_variable,
                self.big_variable
            ]
            for _l in l:
                l_formula = self._variable_rename(_l, l_formula)
            return l_formula

    def _variable_rename(self, variable_list, l_formula):
        tmp = " ".join(l_formula)
        not_used = []
        used = []
        for x in variable_list:
            if x not in tmp:
                not_used.append(x)
            else:
                used.append(x)
        if len(not_used) > 0 and len(used) > 0:
            used_v = random.choice(used)
            new_v = random.choice(not_used)
            for i, f in enumerate(l_formula):
                l_formula[i] = f.replace(used_v, new_v)
        return l_formula

    def variable_replace(self, l_formula):
        if random.random() > self.p:
            return l_formula
        else:
            l = [
                self.x_variable,
                self.greek_variable
            ]
            for _l in l:
                l_formula = self._variable_replace(_l, l_formula)
            return l_formula

    def _variable_replace(self, variable_list, l_formula):
        tmp = " ".join(l_formula)
        used = []
        for x in variable_list:
            if x in tmp:
                used.append(x)
        if len(used) > 0:
            used_v = random.choice(used)
            op = random.choice(self.vr_op)
            num = random.randint(1, 100)
            if op != '\\frac':
                new_v = f'({used_v}{op}{num})'
            else:
                new_v = '\\frac{' + str(used_v)+'}{'+str(num)+'}'
            for i, f in enumerate(l_formula):
                l_formula[i] = f.replace(used_v, new_v)
        return l_formula

    def op_synonym_replace(self, l_formula):
        if random.random() > self.p:
            return l_formula
        else:
            tmp = " ".join(l_formula)
            used = []
            for x in self.op_synonym:
                if x in tmp:
                    used.append(x)
            if len(used) > 0:
                used_v = random.choice(used)
                new_v = random.choice(self.op_synonym[used_v])
                for i, f in enumerate(l_formula):
                    l_formula[i] = f.replace(used_v, new_v)
        return l_formula

    def num_replace(self, l_formula):
        if random.random() > self.p:
            return l_formula
        else:
            tmp = " ".join(l_formula)
            all_num = re.findall('[\d\.]+',tmp)
            if len(all_num) > 0:
                used_num = random.choice(all_num)
                new_num = str(random.randint(1, 9)) + used_num[1:]
                for i, f in enumerate(l_formula):
                    l_formula[i] = f.replace(used_num, new_num)
            return l_formula

    # ------- augmentation for whole question ------------
    def random_shuffle_clause(self, text):
        if random.random() > self.p:
            return text
        else:
            flag_i = [text.find(s) for s in self.sub_ques if s in text]
            if len(flag_i):
                i = min(flag_i)
                clauses = re.split('[,.，。]',text[:i])
                rest = text[i:]
            else:
                clauses = re.split('[,.，。]',text)
                rest = ''
            if len(clauses[-1]) == 0:
                clauses = clauses[:-1]
            if len(clauses) > 1:
                condition = clauses[:-1]
                question = clauses[-1]
                return "，".join(random.sample(condition, len(condition)) + [question]) + rest
            else:
                return text

    def random_clause_insert(self, text):
        if random.random() > self.p:
            return text
        else:
            flag_i = [text.find(s) for s in self.sub_ques if s in text]
            if len(flag_i):
                i = min(flag_i)
                clauses = re.split('[,.，。]',text[:i])
                rest = text[i:]
            else:
                clauses = re.split('[,.，。]',text)
                rest = ''
            if len(clauses[-1]) == 0:
                clauses = clauses[:-1]
            if len(clauses) > 1:
                condition = clauses[:-1]
                question = clauses[-1]
                random_idx = random.randint(0, len(condition)-1)
                repeat = random.choice(condition)
                condition.insert(random_idx, repeat)
                return "，".join(condition + [question]) + rest
            else:
                return text

    def extract_formula(self, text):
        l = []
        f = []
        end = 0
        l = []
        f = []
        for m in self.zh_pattern.finditer(text):
            if m.start() != end:
                if text[end:m.start()] != '.' and text[end:m.start()] != ',':
                    l.append(f"FORMULA{len(f)}")
                    f.append(text[end:m.start()])
                else:
                    l.append(text[end:m.start()])
            l.append(m.group())
            end = m.end()
        if end != len(text):
            l.append(f"FORMULA{len(f)}")
            f.append(text[end:])
        return l, f

    def restore_formula(self, text, l_formula):
        # text: str
        # l_formula: list
        for i in range(len(l_formula)-1, -1, -1):
            text = text.replace(f"FORMULA{i}", l_formula[i])
        return text

    def apply_augment(self, text, method='all'):
        """
        """
        text = text.replace(' ', '')
        # Step1
        l_text, l_formula = self.extract_formula(text)
        if len(l_text) > len(l_formula) and self.args.text_augment:
            text = "".join(l_text)
            if self.pure_zh_pattern.search(text):
                # 包含中文
                # Step2
                if method == 'all':
                    for f in self.text_func_list:
                        text = f(text)
                elif method == 'random':
                    text = random.choice(self.text_func_list)(text)
                else:
                    raise NotImplemented
        # Step3
        if len(l_formula) > 0 and self.args.formula_augment:
            if method == 'all':
                for f in self.formula_func_list:
                    l_formula = f(l_formula)
            elif method == 'random':
                l_formula = random.choice(self.formula_func_list)(l_formula)
            else:
                raise NotImplemented
        # Step4
        if self.args.ques_augment:
            if method == 'all':
                for f in self.ques_func_list:
                    text = f(text)
            elif method == 'random':
                text = random.choice(self.ques_func_list)(text)
            else:
                raise NotImplemented
        # Step5
        return self.restore_formula(text, l_formula)
