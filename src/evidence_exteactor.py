from rank_bm25 import BM25Okapi
from underthesea import sent_tokenize, word_tokenize

class EvidenceExtractor:
    def __init__(self, max_evidence_length=200):
        self.max_evidence_length = max_evidence_length

    def tokenize_vietnamese(self, text):
        """Tokenize Vietnamese text"""
        try:
            return word_tokenize(text.lower(), format="text").split()
        except:
            return text.lower().split()

    def extract_evidence(self, context, claim):
        """Extract best evidence sentence using BM25"""
        if not isinstance(context, str) or not isinstance(claim, str):
            return ""
        sentences = sent_tokenize(context)
        if len(sentences) == 0:
            return ""

        tokenized_sentences = [self.tokenize_vietnamese(sent) for sent in sentences]

        # BM25
        try:
            bm25 = BM25Okapi(tokenized_sentences)
            tokenized_claim = self.tokenize_vietnamese(claim)

            scores = bm25.get_scores(tokenized_claim)

            if len(scores) == 0:
                return sentences[0][:self.max_evidence_length] if sentences else ""

            best_idx = np.argmax(scores)
            best_sentence = sentences[best_idx]

            if len(best_sentence) > self.max_evidence_length:
                best_sentence = best_sentence[:self.max_evidence_length]

            return best_sentence
        except Exception as e:
            print(f"Error in BM25: {e}")
            return sentences[0][:self.max_evidence_length] if sentences else ""
