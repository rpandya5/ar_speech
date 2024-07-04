import requests
from collections import defaultdict, Counter
import re
import os
import sys


class TextPredictor:
    def __init__(self):
        # creating different data structures to store all the word frequencies + patterns
        self.word_freq = Counter()  # how frequently specific words come up
        self.bigram_freq = defaultdict(Counter)  # how frequently pairs
        self.trigram_freq = defaultdict(
            lambda: defaultdict(Counter))  # how frequently 3 words

        self.common_words = set()  # set of predefined common words
        self.common_phrases = []  # list of predefined common phrases
        self.valid_words = set()  # set of words appear 1+ in txt

        self.load_common_words()  # load predef common phrase/words
        self.train_model()  # train model w/ big.txt

    def load_common_words(self):
        # list of common everyday words
        common_words = """the and to a of in is it you that he was for on are with as I his they be at one have this from or had by hot but some what there we can out other were all your when up use word how said an each she which do their time if will way about many then them would write like so these her long make thing see him two has look more day could go come did my sound no most number who over know water than call first people may down side been now find any new work part take get place made live where after back little only round man year came show every good me give our under name very through just form much great think say help low line before turn cause same mean differ move right boy old too does tell sentence set three want air well also play small end put home read hand port large spell add even land here must big high such follow act why ask men change went light kind off need house picture try us again animal point mother world near build self earth father head stand own page should country found answer school grow study still learn plant cover food sun four thought let keep eye never last door between city tree cross since hard start might story saw far sea draw left late run don't while press close night real life few stop walk run jump talk listen speak drive ride write read watch sing dance think feel touch hold open close turn move sit stand stay leave go come stop start begin finish end continue repeat play rest eat drink sleep wake cook bake clean wash brush comb cut paint draw build fix break repair create destroy buy sell give take borrow lend find lose win lose gain spend save earn pay work rest meet greet visit call text email message shout whisper argue agree laugh cry smile frown ask answer tell listen understand know learn teach show hide see look watch observe view examine inspect check count measure weigh mix blend stir shake throw catch pull push lift drop carry drag roll slide bend stretch twist tie untie lock unlock open close break fix clean dirty fill empty light dark hot cold warm cool wet dry loud quiet soft hard heavy light strong weak fast slow quick easy difficult simple complex smooth rough sharp dull straight curved flat round square rectangular triangle circle oval thin thick tall short wide narrow deep shallow high low big small large tiny close near far distant up down left right inside outside above below beside under in out on off at to from with without about over under around through across between among along against before after during since until till by of for against per via despite considering regarding concerning except including excluding considering plus minus times divided equal plus minus multiply divide add subtract equal greater less more fewer lot many some most any none each every other another next last first second third fourth fifth sixth seventh eighth ninth tenth hundred thousand million billion trillion daily weekly monthly yearly hourly minutely secondly day week month year hour minute second morning afternoon evening night today tomorrow yesterday now later soon always never often sometimes rarely occasionally usually frequently infrequently morning noon afternoon evening night midnight dawn dusk daybreak sunrise sunset twilight midnight daynight weekday weekend holiday vacation work school home office shop store restaurant cafe park beach forest mountain river lake ocean sea land air space sky ground underground surface underground above below inside outside left right center front back top bottom edge corner side middle part whole piece bit section slice segment fraction portion unit pair couple group team bunch bundle set series array list collection selection range variety type kind sort category class group division category class section branch field area region zone district territory province state county city town village community neighborhood street road lane avenue drive boulevard highway path track trail route course journey trip travel tour expedition voyage adventure mission quest pursuit search hunt race competition game match sport exercise training practice test exam study research experiment analysis investigation exploration discovery invention creation development design construction production manufacture assembly build repair maintenance service operation management administration organization preparation planning coordination arrangement execution performance delivery communication information data report message news story article book document file record note letter memo email text call phone fax signal sound voice noise music song melody tune rhythm beat harmony chord scale key instrument voice sound noise word sentence phrase expression statement speech talk lecture presentation address comment remark conversation discussion debate argument dialogue interview chat dialogue negotiation consultation meeting conference seminar workshop course lesson class training education teaching instruction tutoring coaching mentoring guidance advice direction suggestion recommendation plan proposal idea opinion view thought belief feeling wish hope dream desire intention aim goal objective purpose target strategy approach method technique tactic way means measure step action activity task job work project assignment duty responsibility role position function operation process procedure practice system program service product application tool device machine equipment material resource supply provision stock store support assistance help aid benefit advantage use value importance significance meaning reason cause source origin background history context circumstance condition factor element feature aspect characteristic property quality attribute nature type sort kind form style shape appearance look design pattern model example sample instance case version variety option choice alternative solution answer key clue hint indication sign signal symptom proof evidence fact detail point matter issue question problem concern difficulty trouble risk danger threat challenge opportunity possibility chance option choice preference priority interest need requirement demand want wish desire expectation hope dream aim goal objective purpose target plan project task job work role position function duty responsibility action activity event incident occasion occurrence experience adventure journey trip travel tour exploration discovery invention creation development innovation improvement progress change transformation transition growth expansion increase rise fall decline decrease reduction loss gain win victory success achievement accomplishment feat performance result outcome effect impact influence consequence reaction response feedback comment remark note mention reference citation quotation example illustration case instance evidence proof support justification explanation reason cause basis foundation root source origin background history story legend myth tale tradition custom practice habit routine behavior conduct attitude manner way style fashion approach method technique system process procedure strategy plan policy rule regulation principle standard criterion measure norm code law order command instruction directive guideline recommendation suggestion advice hint tip clue indication sign signal symptom mark trace print impression footprint fingerprint sign token evidence clue hint indication sign mark print footprint fingerprint track trail scent smell odor aroma fragrance perfume bouquet essence flavor taste savor touch feel texture sound noise voice note tone beat rhythm music song melody tune harmony chord sound voice speech talk conversation dialogue discussion debate argument negotiation consultation meeting conference seminar workshop course lesson class training education teaching instruction learning study practice preparation training experience qualification ability skill talent gift strength weakness advantage disadvantage benefit drawback risk danger opportunity chance possibility potential capacity capability competence proficiency mastery expertise knowledge understanding insight wisdom intelligence smart clever wise bright brilliant genius talented gifted skilled experienced qualified trained educated learned knowledgeable informed aware familiar proficient expert master professional specialist authority source reference authority expert master specialist professional authority reference source information data fact detail point item element feature aspect characteristic property quality attribute nature trait feature aspect detail fact data information knowledge insight understanding wisdom intelligence smart clever bright brilliant wise genius talented gifted skilled experienced qualified trained educated learned knowledgeable informed aware familiar proficient expert master professional specialist authority reference source information data fact detail point item element feature aspect characteristic property quality attribute nature trait feature aspect detail fact data information knowledge insight understanding wisdom intelligence smart clever bright brilliant wise genius talented gifted skilled experienced qualified trained educated learned knowledgeable informed aware familiar proficient expert master professional specialist authority reference source information data fact detail point item element feature aspect characteristic property quality attribute nature trait feature aspect detail fact data information knowledge insight understanding wisdom intelligence smart clever bright brilliant wise genius talented gifted skilled experienced qualified trained educated learned knowledgeable informed aware familiar proficient expert master professional specialist authority reference source information data fact detail point item element feature aspect characteristic property quality attribute nature trait feature aspect detail fact data information knowledge insight understanding wisdom intelligence smart clever bright brilliant wise genius talented gifted skilled experienced qualified trained educated learned knowledgeable informed aware familiar proficient expert master professional specialist authority reference source information data fact detail point item element feature aspect characteristic property quality attribute nature trait feature aspect detail fact data information knowledge insight understanding wisdom intelligence smart clever bright brilliant wise genius talented gifted skilled experienced qualified trained educated learned knowledgeable informed aware familiar proficient expert master professional specialist authority reference source information data fact detail point item element feature aspect characteristic property quality attribute nature trait feature aspect detail fact data information knowledge insight understanding wisdom intelligence smart clever bright brilliant wise genius talented gifted skilled experienced qualified trained educated learned knowledgeable informed aware familiar proficient expert master professional specialist authority reference source information data fact detail point item element feature aspect characteristic property quality attribute nature trait feature aspect detail fact data information knowledge insight understanding wisdom intelligence smart clever bright brilliant wise genius talented gifted skilled experienced qualified trained educated learned knowledgeable informed aware familiar proficient expert master professional specialist authority reference source information data fact detail point item element feature aspect characteristic property quality attribute nature trait feature aspect detail fact data information knowledge insight understanding wisdom intelligence smart clever bright brilliant wise genius talented gifted skilled experienced qualified trained educated learned knowledgeable informed aware familiar proficient expert master professional specialist authority reference source information data fact detail point item element feature aspect characteristic property quality attribute nature trait feature aspect detail fact data information knowledge insight understanding wisdom intelligence smart clever bright brilliant wise genius talented gifted skilled experienced qualified trained educated learned knowledgeable informed aware familiar proficient expert master professional specialist authority reference source information data fact detail point item element feature aspect characteristic property quality attribute nature trait feature aspect detail fact data information knowledge insight understanding wisdom intelligence smart clever bright brilliant wise genius talented gifted skilled experienced qualified trained educated learned knowledgeable informed aware familiar proficient expert master professional specialist authority reference source information data fact detail point item element feature aspect characteristic property quality attribute nature trait feature aspect detail fact data information knowledge insight understanding wisdom intelligence smart clever bright brilliant wise genius talented gifted skilled experienced qualified trained educated learned knowledgeable informed aware familiar proficient expert master professional specialist authority reference source information data fact detail point item element feature aspect characteristic property quality attribute nature trait feature aspect detail fact data information knowledge insight understanding wisdom intelligence smart clever bright brilliant wise genius talented gifted skilled experienced qualified trained educated learned knowledgeable informed aware familiar proficient expert master professional specialist authority reference source information data fact detail point item element feature aspect characteristic property quality attribute nature trait feature
        """.split()
        self.common_words = set(common_words)

        # common phrases
        self.common_phrases = ["can you", "do you", "how are", "what is", "where is", "when is", "why is", "who is", "is it", "are you",
                               "will you", "could you", "would you", "should I", "can I", "may I", "I am", "you are", "he is", "she is", "they are", "we are", "it is"]

    def train_model(self):

        # download big.txt
        if not os.path.exists('big.txt'):
            big_txt_url = "http://norvig.com/big.txt"
            response = requests.get(big_txt_url)
            with open('big.txt', 'w', encoding='utf-8') as f:
                f.write(response.text)

        # read
        with open('big.txt', 'r', encoding='utf-8') as f:
            text = f.read().lower()

        # extracting the words
        words = re.findall(r'\w+', text)

        # counting word frequency
        self.word_freq.update(words)

        # counting frequency pairs (bigram) and thriplets (trigram)
        for w1, w2, w3 in zip(words, words[1:], words[2:]):
            self.bigram_freq[w1][w2] += 1
            self.trigram_freq[w1][w2][w3] += 1

        # set of words that happen 1+
        self.valid_words = set(
            word for word, freq in self.word_freq.items() if freq > 1)

        print("Model trained successfully.")

    def predict(self, context, num_predictions=4):
        words = context.lower().split()
        if not words:  # if there is no context then give common words
            return [context + word for word in self.common_words][:num_predictions]

        current_word = words[-1]
        prev_word = words[-2] if len(words) > 1 else ""
        prev_prev_word = words[-3] if len(words) > 2 else ""

        # check common phrases
        for phrase in self.common_phrases:
            if context.lower().endswith(phrase[:len(context)]) and phrase != context.lower():
                return [context + phrase[len(context):]]

        # predictions based on context
        if prev_prev_word and prev_word:
            predictions = self.trigram_freq[prev_prev_word][prev_word].most_common(
                num_predictions * 4)
        elif prev_word:
            predictions = self.bigram_freq[prev_word].most_common(
                num_predictions * 4)
        else:
            predictions = self.word_freq.most_common(num_predictions * 4)

        # filter predictions
        filtered_predictions = []
        for word, _ in predictions:
            if word.startswith(current_word) and word != current_word:
                if word in self.valid_words:
                    filtered_predictions.append(
                        context[:-len(current_word)] + word)
            if len(filtered_predictions) == num_predictions:
                break

        # if not enough predictions, add common words starting w/ inpiut
        if len(filtered_predictions) < num_predictions:
            for word in self.common_words:
                if word.startswith(current_word) and word != current_word:
                    prediction = context[:-len(current_word)] + word
                    if prediction not in filtered_predictions:
                        filtered_predictions.append(prediction)
                    if len(filtered_predictions) == num_predictions:
                        break

        # if not enough predictions, add word completions
        if len(filtered_predictions) < num_predictions:
            completions = [w for w in self.valid_words if w.startswith(
                current_word) and w != current_word]
            for completion in completions:
                prediction = context[:-len(current_word)] + completion
                if prediction not in filtered_predictions:
                    filtered_predictions.append(prediction)
                if len(filtered_predictions) == num_predictions:
                    break

        # put common words in predictions hgiher up
        boosted_predictions = []
        for pred in filtered_predictions:
            if pred.split()[-1] in self.common_words:
                boosted_predictions.insert(0, pred)
            else:
                boosted_predictions.append(pred)

        return boosted_predictions[:num_predictions] if boosted_predictions else [context] * num_predictions

# reads a single character without needing to hit enter


def getch():
    import sys
    import tty
    import termios
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch


def main():
    predictor = TextPredictor()
    context = ""
    print("Start typing (press Ctrl+C to exit):")

    while True:
        char = getch()  # READ CHARACTER

        if ord(char) == 3:  # Ctrl+C
            print("\nExiting...")
            break

        if char == '\x7f':  # backspace
            if context:
                context = context[:-1]  # remove character from context
                sys.stdout.write('\b \b')  # move cursor back and remove
                sys.stdout.flush()
        else:
            context += char  # add character and show
            sys.stdout.write(char)
            sys.stdout.flush()

        # get predictions
        predictions = predictor.predict(context)

        # clear line and move to begining
        sys.stdout.write('\r' + ' ' * (len(context) + 50) + '\r')
        sys.stdout.write(context)
        sys.stdout.flush()

        # show predictions
        if predictions:
            print(f"\nPredictions: {', '.join(predictions)}")
        else:
            print("\nNo predictions available")

        # cursor back to the end of the context
        sys.stdout.write('\r' + context)
        sys.stdout.flush()


if __name__ == "__main__":
    main()
