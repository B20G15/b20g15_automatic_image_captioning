from fastapi import FastAPI,Request,Form
from fastapi.templating import Jinja2Templates
app = FastAPI()
templates = Jinja2Templates(directory = 'templates/')
from sklearn.datasets import load_iris
iris = load_iris()
import pandas as pd
df = pd.DataFrame(data = iris.data,columns = iris.feature_names)
df['Species'] = iris.target
#print(df.head())
X = df.iloc[:,0:4]
y = df.iloc[:,4]
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state = 42)
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(X_train,y_train)

#from sklearn.metrics import accuracy_score
#print('Accuracy' ,accuracy_score(y_test,y_pred))
@app.get('/')
def read_form():
    return 'hello world'
@app.get('/test')
def form_post(request:Request):
    res = '{{<Caption>}}'
    #return templates.TemplateResponse('test.html',context  = {'request':request,'result':res})
    return templates.TemplateResponse('test_image_caption.html',context  = {'request':request,'result':res})
@app.post('/test')
def form_post(request:Request,ImageID:int=Form(...)):
        #result = dt.predict([[ImageID]])
        result = ImageID
        t = 'Caption_Sample'
        pic = 'https://cdn.pixabay.com/photo/2017/05/24/08/22/iris-2339883_1280.jpg'
        #pic = 'path_of_input_ImageID.jpg file'
        return templates.TemplateResponse('test_image_caption.html',context = {'request':request,'result':t,'ImageID':ImageID,'pic':pic})

   



