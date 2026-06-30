# ============================================================
# QSkill AI Internship - Task 2: Spam Mail Detector
# Domain : Artificial Intelligence & Machine Learning
# June 2026
# ============================================================

import re
import string
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc
)

# -- 1. Load Dataset (built-in SMS Spam collection) -----------
# Using a robust inline sample that mirrors the real SMS Spam Collection
# structure (tab-separated: label \t message)
SAMPLE_DATA = """ham\tGo until jurong point, crazy.. Available only in bugis n great world la e buffet... Cine there got amore wat...
spam\tFree entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18's
ham\tU dun say so early hor... U c already then say...
ham\tNah I don't think he goes to usf, he lives around here though
spam\tFreeMsg Hey there darling it's been 3 week's now and no word back! I'd like some fun you up for it still? Tb ok! XxX std chgs to send, 1.50 to rcv
ham\tEven my brother is not like to speak with me. They treat me like aids patent.
ham\tAs per your request 'Melle Melle (Oru Minnaminunginte Nurungu Vettam)' has been set as your callertune for all Callers. Press *9 to copy your friends Callertune
spam\tWINNER!! As a valued network customer you have been selected to receivea GBP900 prize reward! To claim call 09061701461. Claim code KL341. Valid 12 hours only.
ham\tHave you ever seriously considered doing something? You know, that thing you said you were gonna do? Well...
spam\tHad your mobile 11 months or more? U R entitled to Update to the latest colour mobiles with camera for Free! Call The Mobile Update Co FREE on 08002986030
ham\tI'm gonna be home soon and i don't want to talk about this stuff anymore tonight, k? I've cried enough today.
spam\tSIX chances to win CASH! From 100 to 20,000 pounds txt> CSH11 and send to 87575. Cost 150p/day, 6days, 16+ TsandCs apply Reply HL 4 info
ham\tBORED IN WORK, BORING WORKMATES! JUST WANNA CHAT? ME TOO! LAG ON, N LETS CHAT. STA- 7908 3A NETWORK. FOR HLP CALL 08000930705.
spam\tARGOS CALLING! for FREE, your secret prize is waiting 4 u at http://www.argosmobile.com. Win GBP500 cash or a GBP5 voucher or call 08454075529
ham\tWinnermen. Good to know you are well. How about you coming to visit some time?
ham\tI forgot 2 ask u all smth.. There's a card game 2nite @ 8. I'll give you the address if you're interested
spam\tCONGRATULATIONS! Thanks to a special offer you are selected to receive one FREE ringtone! Reply YES to claim your FREE ringtone!
ham\tSometimes I'll start a sentence and I don't even know where it's going. I just hope I find it along the way.
spam\tOur records indicate that your recent trip has earned you a GBP1500 voucher. Call 0905 872 0576 to claim. Operator: IBILL PO BOX 1400 WA15 8XT 150 ppm
ham\tSorry, I'll call later in the morning, if I can.
spam\tSend a message or call 08712400200 to get your FREE CD. Only mobile users can claim this prize. CALL NOW!
ham\tCould you tell me what all you did last weekend?
ham\tOk so what does that make me? ... Never mind don't answer that.
spam\tGet the powerful Lotto WINNING system here http://tinyurl.com/k8q7x6 USE code RED50 to get 50% off! WIN the Lottery JACKPOT NOW!
ham\tI've been searching for the right words to thank you for this breather. I promise i wont take your help for granted and will fulfil my promise.
ham\tFree msg: no need to pay now - get your free message from mates. Retrieve your free ringtone at http://www.freemobiles.biz
ham\tIf only we could get them interested it would be helpful, because they could do it in their sleep and probably make some money too.
spam\tYour GBP1,000 prize awarded today, draw ref 11/07/26. Call 09061702893 and quote ref 1121 to collect. Lines open Mon-Sun 9am-10pm Cost 10p per min
ham\tToday is 8th May. I have to say this is a crazy month. Two major accidents in a week!
spam\tText me and receive hot girls for FREE! Msg HotGirls to 80804 and receive pictures of real hot girls who want to meet you.
ham\tWell what about the argument you had with your sister about school?
ham\tActually I was sleeping and didn't see your message.
ham\tActually i thought it was good to have a male gynaecologist examine you... I think we misunderstood each other
spam\tYOUR CHANCE TO WIN PRIZES TODAY! CALL 0906 6461 0468. CLAIM CODE: B17526 NETWORK OPERATOR CALL COST 150P PER MIN.
ham\tAm stuck in traffic. Will be home in 30 min.
spam\tTODAY'S OFFER: WIN GBP5,000 NOW! Call 09064012160 and give your Security Number 55 now to collect your prize! Claim code: brs-3 T&C at www.tdnewsletter.com
ham\tWhatever you decide will be fine with me.
ham\tI'll have to talk to you about it later.
spam\tFree ringtone waiitng to be collected. Simply text the password \"MIX\" to 85069 to verify. Get Usher ringtone FREE. Txts@2.39 ea,T&Cs www.ldew.com/1stop
ham\tI don't know what's wrong with me. I feel empty.
ham\tSurprised? Me too. How's your day going? Call me when you get a chance ok?
spam\tGREAT NEWS! You have been SELECTED to receive a GBP2000 prize reward! Call 09064019788 NOW! T&C's apply.
ham\tTake it slow and breathe, think about why things went the way they did.
ham\tYes, but I also like them, especially in rainy weather.
spam\tNotice: You have WON a guaranteed GBP1000 cash or GBP5000 prize. Call 09061743386. T&C'S apply. The claims line closes at 6pm. Cost 10p per min.
ham\tLook at the size of them!!
ham\tI just felt like telling you the whole of today's trip, to make you feel how I'm feeling now.
spam\tYou have an important customer service announcement. CALL 08702490080 NOW! 1 New Msg. Last week your voicemail got 1 new msg. Msg: Please call back.
ham\tOh. What's wrong with you?
spam\tSuper chance! Win GBP2000 IN CASH EVERY WEEK for a YEAR! All you have to do is reply YES to this message. Claim CODE: LUCKY2. T&Cs apply.
ham\tOh no, no worries. I was just going to say that I'm not doing very well.
ham\tSo do you think I should come over there?
spam\tGo to GAME.CO.UK for your chance to WIN. Text MATCH to 83750 to win a GBP1000 prize. Operator: Txtsurf, 87115. 150p/txt. T&Cs apply.
ham\tI'll be there in a couple hours.
ham\tI just finished eating. Wait for me in the lobby.
spam\tCLAIM A FREE MOBILE EVERY MONTH! Text CLAIM to 8552. Cost 150p/wk. To stop, text STOP to 8552. T&Cs at www.mcompete.com.
ham\tI had a great time talking to you. Let's do it again soon.
ham\tSorry, I meant to say I'll be late. Give me 10 more minutes.
spam\tWin a GBP1000 cash prize or a prize worth GBP5000. CALL 09063131030. T and C's apply. Calls cost 10p per min. Send STOP to 82324 to stop messages.
ham\tOk I'll text you when I'm on my way.
ham\tYou know what I actually thought the same, that is why I was bothered by it.
ham\tFrom where have you ordered it?
spam\tMOBILE QUIZ CLUB. Your CHANCE to WIN GBP100 of Amazon Vouchers! CALL 09066358152 NOW. Costs 150p/min to call. T&Cs at www.mobilequizclub.co.uk
ham\tIt's all about the journey, not the destination.
ham\tThings are going okay here. How are things with you?
spam\tYou're a WINNER in our monthly prize draw! Collect your special reward: Call 09061743386 T&Cs: apply Calls 10p/min.\t
ham\tI'm not so sure whether to be happy or sad about today.
ham\tOk just finished getting dressed. I'll be right over.
spam\tAmazing! SMS this week only! Send KISS to 95050 and collect 2 amazing..., by sending this, you agree to T&Cs at www.e-promotions.co.uk/wap 150p/msg
ham\tPlease tell me your plan to get here on time.
ham\tCall me when you wake up ok? I need to talk to you.
spam\tUrgent! You have won a 1 week FREE membership in our GBP100,000 Prize Jackpot! Txt the word: CLAIM to No: 81010 T&C www.dbuk.net LCCLTD POBOX 4403LDNW1A7RW18
ham\tI know it's hard but it'll get better.
ham\tThis is the number you need to call to reach me.
spam\tYou have been selected to WIN a GBP500 Asda Gift Card. To claim your prize, call 0901 6574945 from your mobile or landline today! Cost 60p/min.
ham\tJust let me know when you're ready.
spam\tClaim your FREE text message prize. Text COMP to 87066 now! Entry cost 50p. T&Cs apply. 16+
ham\tI'll come by around 7. See you then.
ham\tI appreciate you being there for me.
spam\tGreat news! Samsung S7 WINNER 4 2day draw. Call 09050003091 to get prize of 3 Hundred quid. You got this.
ham\tWow, that's awesome! When did you get there?"""

# parse the inline data
rows = []
for line in SAMPLE_DATA.strip().split('\n'):
    if '\t' in line:
        label, msg = line.split('\t', 1)
        rows.append({'label': label.strip(), 'message': msg.strip()})

df = pd.DataFrame(rows)
print("=" * 55)
print("           SPAM MAIL DETECTOR")
print("=" * 55)
print(f"\nDataset shape   : {df.shape}")
print(f"\nLabel distribution:\n{df['label'].value_counts()}")
print(f"\nSpam ratio  : {df['label'].value_counts(normalize=True)['spam']*100:.1f}%")

# -- 2. EDA ----------------------------------------------------
df['message_length'] = df['message'].apply(len)
df['word_count']     = df['message'].apply(lambda x: len(x.split()))

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle("Spam vs Ham - Message Analysis", fontsize=13, fontweight='bold')

for label, color in [('ham', '#4C72B0'), ('spam', '#DD8452')]:
    subset = df[df['label'] == label]
    axes[0].hist(subset['message_length'], alpha=0.65, label=label, color=color, bins=20, edgecolor='white')
    axes[1].hist(subset['word_count'],     alpha=0.65, label=label, color=color, bins=20, edgecolor='white')

axes[0].set_title("Message Length Distribution")
axes[0].set_xlabel("Characters"); axes[0].legend()
axes[1].set_title("Word Count Distribution")
axes[1].set_xlabel("Words"); axes[1].legend()

plt.tight_layout()
plt.savefig("spam_eda.png", dpi=150, bbox_inches='tight')
plt.close()

print("\n[OK] EDA charts saved.")

# -- 3. Text Preprocessing -------------------------------------
STOPWORDS = {
    'i','me','my','myself','we','our','ours','ourselves','you',"you're",
    "you've","you'll","you'd",'your','yours','yourself','he','him','his',
    'himself','she',"she's",'her','hers','herself','it',"it's",'its',
    'itself','they','them','their','theirs','themselves','what','which',
    'who','whom','this','that',"that'll",'these','those','am','is','are',
    'was','were','be','been','being','have','has','had','having','do',
    'does','did','doing','a','an','the','and','but','if','or','because',
    'as','until','while','of','at','by','for','with','about','against',
    'between','into','through','during','before','after','above','below',
    'to','from','up','down','in','out','on','off','over','under','again',
    'further','then','once','here','there','when','where','why','how',
    'all','both','each','few','more','most','other','some','such','no',
    'nor','not','only','own','same','so','than','too','very','s','t',
    'can','will','just','don',"don't",'should',"should've",'now','d',
    'll','m','o','re','ve','y','ain',"aren't","couldn't","didn't",
    "doesn't","hadn't","hasn't","haven't","isn't","mightn't","mustn't",
    "needn't","shan't","shouldn't","wasn't","weren't","won't","wouldn't"
}

def preprocess(text: str) -> str:
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', 'url', text)
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = [w for w in text.split() if w not in STOPWORDS and len(w) > 1]
    return ' '.join(tokens)

df['clean_message'] = df['message'].apply(preprocess)
print("\nSample preprocessing:")
for _, row in df[df['label'] == 'spam'].head(2).iterrows():
    print(f"  Original : {row['message'][:70]}...")
    print(f"  Cleaned  : {row['clean_message'][:70]}...\n")

# -- 4. Feature Extraction (TF-IDF) ---------------------------
tfidf = TfidfVectorizer(max_features=3000, ngram_range=(1, 2))
X = tfidf.fit_transform(df['clean_message'])
y = (df['label'] == 'spam').astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train : {X_train.shape[0]} | Test : {X_test.shape[0]}")
print(f"TF-IDF features : {X_train.shape[1]:,}")

# -- 5. Train Models -------------------------------------------
models = {
    "Naive Bayes"        : MultinomialNB(alpha=0.1),
    "Logistic Regression": LogisticRegression(max_iter=300, C=5.0, solver='liblinear', random_state=42),
}

results = {}
print("\n" + "=" * 55)
print("              MODEL EVALUATION")
print("=" * 55)

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred      = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]

    results[name] = {
        'model'     : model,
        'y_pred'    : y_pred,
        'y_prob'    : y_pred_prob,
        'accuracy'  : accuracy_score(y_test, y_pred),
        'precision' : precision_score(y_test, y_pred),
        'recall'    : recall_score(y_test, y_pred),
        'f1'        : f1_score(y_test, y_pred),
    }

    print(f"\n-- {name}")
    print(f"   Accuracy  : {results[name]['accuracy']:.4f}")
    print(f"   Precision : {results[name]['precision']:.4f}")
    print(f"   Recall    : {results[name]['recall']:.4f}")
    print(f"   F1 Score  : {results[name]['f1']:.4f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['Ham','Spam'])}")

# -- 6. Confusion Matrices + ROC ------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle("Spam Detector - Model Evaluation", fontsize=13, fontweight='bold')

for ax, (name, res) in zip(axes[:2], results.items()):
    cm = confusion_matrix(y_test, res['y_pred'])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Ham', 'Spam'])
    disp.plot(ax=ax, colorbar=False, cmap='Blues')
    ax.set_title(f"Confusion Matrix\n{name}", fontsize=9)

# ROC curves
ax = axes[2]
for name, res in results.items():
    fpr, tpr, _ = roc_curve(y_test, res['y_prob'])
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, label=f"{name.split()[0]} (AUC={roc_auc:.2f})", linewidth=2)

ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve"); ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig("spam_evaluation.png", dpi=150, bbox_inches='tight')
plt.close()

# -- 7. Metrics Comparison Bar --------------------------------
metrics = ['accuracy', 'precision', 'recall', 'f1']
fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(len(metrics))
width = 0.35
colors_m = ['#4C72B0', '#DD8452']

for i, (name, res) in enumerate(results.items()):
    vals = [res[m] for m in metrics]
    bars = ax.bar(x + (i - 0.5) * width, vals, width, label=name,
                  color=colors_m[i], alpha=0.85)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=8)

ax.set_xticks(x)
ax.set_xticklabels([m.capitalize() for m in metrics])
ax.set_ylim(0.6, 1.1)
ax.set_title("Spam Detector - Metrics Comparison")
ax.set_ylabel("Score")
ax.legend()
plt.tight_layout()
plt.savefig("spam_metrics_comparison.png", dpi=150, bbox_inches='tight')
plt.close()

# -- 8. Live Prediction Demo -----------------------------------
best_model_name = max(results, key=lambda n: results[n]['f1'])
best_model      = results[best_model_name]['model']

test_messages = [
    "Congratulations! You've won a GBP1000 prize. Call 08712345678 NOW to claim!",
    "Hey, are you free this evening? Let's grab dinner.",
    "FREE entry to win GBP500 cash! Text WIN to 87575. Cost 50p/msg. T&Cs apply.",
    "I'll be at the office by 9. Can you send me the report before that?",
    "URGENT! Your account is suspended. Verify details at http://secure-bank.xyz now!",
]

print("\n" + "=" * 55)
print("          LIVE PREDICTION DEMO")
print("=" * 55)
print(f"Model: {best_model_name}\n")

for msg in test_messages:
    cleaned = preprocess(msg)
    feat    = tfidf.transform([cleaned])
    pred    = best_model.predict(feat)[0]
    prob    = best_model.predict_proba(feat)[0][1]
    label   = "[SPAM]" if pred == 1 else "[HAM]"
    print(f"  {label}  (conf: {prob:.2%})")
    print(f"  \"{msg[:65]}...\"" if len(msg) > 65 else f"  \"{msg}\"")
    print()

# -- 9. Summary ------------------------------------------------
print("=" * 55)
print("SUMMARY")
print("=" * 55)
for name, res in results.items():
    print(f"  {name:<25} F1: {res['f1']:.4f}  Acc: {res['accuracy']:.4f}")

print(f"\n* Best Model : {best_model_name}")
print(f"  F1 Score   : {results[best_model_name]['f1']:.4f}")
print("\n[OK] All outputs saved to current directory.")
