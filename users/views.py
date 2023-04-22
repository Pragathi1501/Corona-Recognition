from django.shortcuts import render, HttpResponse
from .forms import UserRegistrationForm
from .models import UserRegistrationModel, CoronaDischargeModel
from django.contrib import messages
from django.core.files.storage import FileSystemStorage
from .rgbglhcodes.StartImagePreprocessing import StartProcess
from .rgbglhcodes.SVMCode import UserSVMCode
from .rgbglhcodes.ShapeUtility import UserImageShape
from .rgbglhcodes.UserBrightness import UserImageBrightness
from django_pandas.io import  read_frame
# Create your views here.
import matplotlib
#matplotlib.use("Agg")
from matplotlib import style
#style.use("ggplot")


# Create your views here.
def UserRegisterActions(request):
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            print('Data is Valid')
            form.save()
            messages.success(request, 'You have been successfully registered')
            form = UserRegistrationForm()
            return render(request, 'UserRegistrations.html', {'form': form})
        else:
            messages.success(request, 'Email or Mobile Already Existed')
            print("Invalid form")
    else:
        form = UserRegistrationForm()
    return render(request, 'UserRegistrations.html', {'form': form})
def UserLoginCheck(request):
    if request.method == "POST":
        loginid = request.POST.get('loginname')
        pswd = request.POST.get('pswd')
        print("Login ID = ", loginid, ' Password = ', pswd)
        try:
            check = UserRegistrationModel.objects.get(loginid=loginid, password=pswd)
            status = check.status
            print('Status is = ', status)
            if status == "activated":
                request.session['id'] = check.id
                request.session['loggeduser'] = check.name
                request.session['loginid'] = loginid
                request.session['email'] = check.email
                print("User id At", check.id, status)
                return render(request, 'users/UserHome.html', {})
            else:
                messages.success(request, 'Your Account Not at activated')
                return render(request, 'UserLogin.html')
        except Exception as e:
            print('Exception is ', str(e))
            pass
        messages.success(request, 'Invalid Login id and password')
    return render(request, 'UserLogin.html', {})
def UserHome(request):
    return render(request, 'users/UserHome.html', {})

def UploadImageForm(request):
    loginid = request.session['loginid']
    data = CoronaDischargeModel.objects.filter(loginid=loginid)
    return render(request, 'users/UserImageUploadForm.html', {'data': data})

def UploadImageAction(request):
    image_file = request.FILES['file']
    # let's check if it is a csv file
    if not image_file.name.endswith('.jpg'):
        messages.error(request, 'THIS IS NOT A JPG  FILE')
    fs = FileSystemStorage(location="media/datasets/")
    filename = fs.save(image_file.name, image_file)
    # detect_filename = fs.save(image_file.name, image_file)
    uploaded_file_url = "/media/datasets/"+filename #fs.url(filename)
    print("Image path ",uploaded_file_url)
    username = request.session['loggeduser']
    loginid = request.session['loginid']
    email = request.session['email']
    obj = StartProcess()
    colorinfo,picbrightness,picshape = obj.process(filename)
    colorinfo = colorinfo.tolist()
    redColor = colorinfo[0]
    greenColor = colorinfo[1]
    blueColor = colorinfo[2]
    picHeight = picshape[0]
    picWidht = picshape[1]
    blockofPixel = picshape[2]
    picbrightness = picbrightness

    CoronaDischargeModel.objects.create(username=username,email=email,loginid=loginid,filename=filename,file=uploaded_file_url,redColor=redColor, greenColor=greenColor, blueColor=blueColor, picHeight=picHeight,picWidht=picWidht, blockofPixel=blockofPixel, picbrightness=picbrightness)
    data = CoronaDischargeModel.objects.filter(loginid=loginid)
    return render(request, 'users/UserImageUploadForm.html', {'data':data})

def UserSVMTest(request):
    obj = UserSVMCode()
    svmMrmse = obj.startSvm()
    knnmrmse = obj.startKnn()
    dtmrmse = obj.startDecisionTree()
    slpmrmse = obj.startSLP()
    return render(request,'users/UserColors.html',{'svmMrmse':svmMrmse,'knnmrmse':knnmrmse,"dtmrmse":dtmrmse,"slpmrmse":slpmrmse})

def UserShapeTest(request):
    data = CoronaDischargeModel.objects.all()
    df = read_frame(data)
    df = df[['picHeight','picWidht','blockofPixel','picbrightness']]
    #print(df.head())
    obj = UserImageShape()
    svmMrmse = obj.startSvm(df)
    knnmrmse = obj.startKnn(df)
    dtmrmse = obj.startDecisionTree(df)
    slpmrmse = obj.startSLP(df)
    return render(request, 'users/UserShapeAsFeatures.html',
                  {'svmMrmse': svmMrmse, 'knnmrmse': knnmrmse, "dtmrmse": dtmrmse, "slpmrmse": slpmrmse})


def UserBrightness(request):
    data = CoronaDischargeModel.objects.all()
    df = read_frame(data)
    df = df[['redColor', 'greenColor', 'blueColor', 'picbrightness']]
    obj = UserImageBrightness()
    svmMrmse = obj.startSvm(df)
    knnmrmse = obj.startKnn(df)
    dtmrmse = obj.startDecisionTree(df)
    slpmrmse = obj.startSLP(df)
    return render(request, 'users/UserBrightnessFeatures.html',
                  {'svmMrmse': svmMrmse, 'knnmrmse': knnmrmse, "dtmrmse": dtmrmse, "slpmrmse": slpmrmse})

    return HttpResponse("This is Shit")

def GetImageHOGRGBGLH(request):
    if request.method=='GET':
        imgname = request.GET.get('imagename')
        print("Values Must Be ",imgname)
        obj = StartProcess()
        colorinfo, picbrightness, picshape = obj.process(imgname)

    loginid = request.session['loginid']
    data = CoronaDischargeModel.objects.filter(loginid=loginid)
    return render(request, 'users/UserImageUploadForm.html', {'data': data})