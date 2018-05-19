import csv

from nltk.corpus import wordnet as wn
import nltk


class WordNetDepth:
    def __init__(self, language):
        self.sentences = {}
        self.sentences_pos = {}
        self.language = language

    def get_wordnet_pos(self, treebank_tag):
        """
        return WORDNET POS compliance to WORDENT lemmatization (a,n,r,v)
        """
        if treebank_tag.startswith('J'):
            return wn.ADJ
        elif treebank_tag.startswith('V'):
            return wn.VERB
        elif treebank_tag.startswith('N'):
            return wn.NOUN
        elif treebank_tag.startswith('R'):
            return wn.ADV
        else:
            # As default pos in lemmatization is Noun
            return wn.NOUN

    def get_sentences(self):
        english = self.language[0]
        spanish = self.language[1]
        data_english = self.read_dataset("../datasets/{}/{}_Train.tsv".format(english, english.capitalize()))
        # data_spanish = Dataset(spanish)
        sentences = []
        for sent in data_english:
            sentences.append(sent['hit_id'])
            self.sentences[sent['hit_id']] = sent['sentence']
        print(len(sentences), len(set(sentences)), len(self.sentences))
        print(set(sentences))
        print(self.sentences)

    def get_words_pos(self):
        for key, val in self.sentences.items():
            self.sentences_pos[key] = nltk.pos_tag(nltk.word_tokenize(val))

    def read_dataset(self, file_path):
        with open(file_path) as file:
            fieldnames = ['hit_id', 'sentence', 'start_offset', 'end_offset', 'target_word', 'native_annots',
                          'nonnative_annots', 'native_complex', 'nonnative_complex', 'gold_label', 'gold_prob']
            reader = csv.DictReader(file, fieldnames=fieldnames, delimiter='\t')

            dataset = [sent for sent in reader]

        return dataset

    def get_depth(self, word, pos):
        word_synset = wn.synsets(word)
        for wordtemp in word_synset:
            wordname = wordtemp.name()
            jword = wordname.split(".")
            for w in jword:
                if (w[0] == word and w[1] == pos):
                    return w[2]
        return 0


language1 = 'english'
language2 = 'spanish'
test = WordNetDepth([language1, language2])
test.get_sentences()
test.get_words_pos()

print(test.sentences_pos)

for key, val in test.sentences_pos.items():
    for word in val:
        print(word[0], word[1])
