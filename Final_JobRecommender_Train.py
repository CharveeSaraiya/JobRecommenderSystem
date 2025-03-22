import pandas as pd
from rake_nltk import Rake
import operator
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')
import wordcloud as w
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,make_scorer,classification_report,accuracy_score
import scikitplot as skplt
from sklearn import svm
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import roc_curve, roc_auc_score
import warnings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import pickle



job_postings = 'DATASET\dice_com-job_us_sample.csv'
df_job = pd.read_csv(job_postings)


# Preview the first 5 lines of the loaded data
df_job.head()

print(df_job.columns)

# Remove postdate, shift, sitename
df_job.drop('postdate', inplace=True, axis=1)
df_job.drop('shift',inplace=True, axis=1)
df_job.drop('site_name', inplace=True, axis =1)
df_job.drop('uniq_id', inplace=True, axis =1)
df_job.drop('jobid', inplace=True, axis =1)

# Preview the first 5 Lines of the loaded data
df_job.head()

df_job.drop_duplicates()
df_job.head(5)

#check to see if we have any missing values at all.
print(df_job.isnull ().values.any ())
# since the above results is true, check number of missing values in each row
df_job.isna().sum()

#drop all rows that have any NaN values
df_job.dropna(inplace=True)

##check that there is no more na
df_job.isna().sum()

# Get index of all rows in skills that contains the value "Null"
jobsNull = df_job[df_job["skills"]=="Null"]. index
# Get index of all rows in skills that contains the value "Please see job description"
jobsDesc1 = df_job[df_job["skills"]=="Please see job description"].index
# Get index of all rows in skills that contains the value (See Job Description)
jobsDesc2 = df_job[df_job["skills"]==" (See Job Description)"]. index
# Get index of all rows in skills that contains the value "SEE BELOW*
jobsDesc3 = df_job[df_job["skills"]=="SEE BELOW"]. index
# Get index of all rows in skills that contains the value "Telecommuting not available Travel not required"
jobsDesc4 = df_job[df_job["skills"]=="Telecommuting not available Travel not required"]. index
# Get index of all rows in skills that contains the value "Refer to Job Description"
jobsDesc5 = df_job[df_job["skills"]=="Refer to Job Description"].index
# Get index of all rows in skills that contains the value "Please see Required Skills"
jobsDesc6 = df_job[df_job["skills"]=="Please see Required Skills"].index
# drop rows of index
df_job.drop(jobsNull, inplace=True)
df_job.drop(jobsDesc1, inplace=True)
df_job.drop(jobsDesc2, inplace=True)
df_job.drop(jobsDesc3, inplace=True)
df_job.drop(jobsDesc4, inplace=True)
df_job.drop(jobsDesc5, inplace=True)
df_job.drop(jobsDesc6, inplace=True)

#Print info of Dataframe
df_job.info()

job = []
stopwordsList = []
cleanJobs = []
# Get the stopwords and store in list
with open("DATASET/stopwords.txt", 'r', encoding="utf-8") as f:
    for word in f:
        word = word.split('\n')
        stopwordsList.append(word[0])

# Tokenizing and Removing stop words from jobtitle

from nltk.tokenize import word_tokenize
import nltk

# Convert all words to lower case and change the shortform
for i in df_job['jobtitle'].values:
  jobs = i. lower ()
  jobs = jobs.replace("QA", "Quality Assurance")
  jobs = jobs.replace("sr", "Senior")
  jobs = jobs.replace("jr", "Junior")
  jobs = jobs.replace("qm", "Quality Manager")
  job.append (jobs)

# tokenize and remove the words from stop words
for j in job:
    text_tokens = word_tokenize(j)
    tokens_without_sw = [f for f in text_tokens if not f in stopwordsList]
    cleanJobs .append(' '.join(tokens_without_sw))

# Add the newly cleaned job title into the dataframe
df_job['clean_jobtitle'] = cleanJobs
df_job.head(5)

# Get the Top 5 Jobs
qty = df_job["clean_jobtitle"].value_counts()[:5].tolist ()
label = df_job["clean_jobtitle"].value_counts () [:5]. index. tolist ()
print(qty)
print("Top 5 Popular Jobs :" + str(label))


#TFIDF
vectorizer = TfidfVectorizer ()
X = vectorizer.fit_transform(df_job['clean_jobtitle'].values)
print(X.shape)
analyze = vectorizer.build_analyzer()
# print( 'Job titles', analyze(str(jobtitle)))
# print ('Document transform', X. toarray ())
# print(vectorizer.get_feature_names())
features = vectorizer.get_feature_names_out()
# indices = zip(*X.nonzero)
# for row, column in indices:
#   print(' (%d, %s) %f' %(row, features [column], X[row, column]))


from sklearn.cluster import KMeans
# Using the elbow method to find the optimal number of clusters
# Within Cluster Sum of Squares(WCSS)
wcss =[]
for i in range (1,15):
    kmeans = KMeans(n_clusters = i,init='k-means++',random_state = 42,max_iter = 600,n_init = 1)
    kmeans.fit(X)
    #inertia method returns wcss for the model
    wcss.append(kmeans.inertia_)

true_k = 7  # Set the desired number of clusters
model = KMeans(n_clusters=true_k, init='k-means++', max_iter=600, n_init=1, random_state=42)
pred = model.fit_predict(X)
order_centeroids = model.cluster_centers_.argsort()[:,::-1]
terms = vectorizer.get_feature_names_out()



label = []
for i in df_job['clean_jobtitle'].values:
      vec = vectorizer.transform([i])
      pred = model.predict (vec)
      if pred == 0:
          label.append("Project Management")
      elif pred == 1:
          label.append("Frontend")
      elif pred == 2:
          label.append("Devops/Software Engineer")
      elif pred == 3:
          label.append("Business Solution Consultant")
      elif pred == 4:
          label.append("Cloud Architect/Network")
      elif pred == 5:
          label.append("Analyst")
      else:
          label.append("IT Business Management")

df_job['Label'] = label
df_job.head(5)

jobSkills = []
for i in df_job['skills']:
    jobSkills.append(i.lower ())

Xclass = vectorizer.fit_transform(jobSkills)
#Split data into test and train. Test size 20% Train Size 80%
X_train,X_test,y_train,y_test = train_test_split(Xclass, label, test_size=0.2, random_state=42)

from sklearn import svm
Csvm = [0.1,0.5,0.8,1,1.5, 2,2.5,3,3.5]
trainAcc = []
testAcc = []
for c in Csvm:
    modelsvm = svm.SVC(C= c, gamma= 1, kernel = 'rbf')
    svmfit = modelsvm.fit(X_train,y_train)
    Y_predTrain = modelsvm.predict(X_train)
    Y_predTest = modelsvm.predict (X_test)
    trainAcc.append(accuracy_score(y_train, Y_predTrain))
    testAcc.append(accuracy_score(y_test, Y_predTest))

plt.plot(Csvm, trainAcc,'ro-', Csvm, testAcc, 'bv--')
plt. legend(['Training Accuracy','Test Accuracy'])
plt.xlabel('Number of C')
plt.ylabel('Accuracy')

svm = svm.SVC(C= 5, gamma= 1, kernel= 'rbf' ,probability=True)
svmfit = svm.fit(X_train,y_train)
svm_predictions = svmfit.predict (X_test)
svm_acc = accuracy_score(y_test,svm_predictions)
print("Accuracy of SVM:" + str(svm_acc))
print(classification_report (y_test, svm_predictions))


# Save the model and vectorizer to files
with open('svm_model.pkl', 'wb') as model_file:
    pickle.dump(svm, model_file)

with open('tfidf_vectorizer.pkl', 'wb') as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)
