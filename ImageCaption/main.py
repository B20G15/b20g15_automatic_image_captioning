from fastapi import FastAPI,Request,Form
from fastapi.templating import Jinja2Templates

app = FastAPI()
templates = Jinja2Templates(directory = 'templates/')
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()

@app.get('/')
def read_form():
    return 'hello world'
@app.get('/test')
def form_post(request:Request):
    res = '{{<Caption>}}'
    #return templates.TemplateResponse('test.html',context  = {'request':request,'result':res})
    return templates.TemplateResponse('test.html',context  = {'request':request,'result':res})
@app.post('/test')
def form_post(request:Request,ImageID:int=Form(...)):
        #result = dt.predict([[ImageID]])
        result = ImageID
        t = 'Caption_Sample'
        pic = 'https://cdn.pixabay.com/photo/2017/05/24/08/22/iris-2339883_1280.jpg'
        #pic = 'path_of_input_ImageID.jpg file'
        return templates.TemplateResponse('test.html',context = {'request':request,'result':t,'ImageID':ImageID,'pic':pic})

   



