from django.shortcuts import render,redirect
import cv2
from django.http import StreamingHttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
import base64
from collections import deque
from .models import Embeds, Profile
from django.contrib.auth.models import User, auth
from django.contrib.auth import authenticate,login,logout
from django.contrib import messages
from deepface import DeepFace
from django.contrib.auth.decorators import login_required
import random as rd
import dlib
import numpy as np
import mediapipe as mp





cap1 = ""
cap2 = ""
recent_images = []
reg_embeds = []
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml') 
name = ""
modwl = DeepFace.build_model('Facenet')
gestures_list = ['Mouth Open','Peace Sign','Open Hand','Fist','Thumbs Up','Thumbs Down','Pointing']
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(r"shape_predictor_68_face_landmarks.dat")
ges_count = 4
gesture = ""
sno = 5
to_verify = []
name_list=  []
id_list = []




def get_landmarks(image):
    rects = detector(image, 1)
    if len(rects) == 0:
        return None
    return np.array([[p.x, p.y] for p in predictor(image, rects[0]).parts()])


mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils


def extract_features(hand_landmarks):
    features = []
    for landmark in hand_landmarks.landmark:
        features.extend([landmark.x, landmark.y, landmark.z])
    return np.array(features)



def recognize_gesture(features):
    if len(features) == 0:
        return "No Gesture"
    
    # Thumb
    thumb_tip_y = features[mp_hands.HandLandmark.THUMB_TIP * 3 + 1]
    thumb_mcp_y = features[mp_hands.HandLandmark.THUMB_MCP * 3 + 1]
    thumb_ip_y = features[mp_hands.HandLandmark.THUMB_IP * 3 + 1]

    # Index
    index_tip_y = features[mp_hands.HandLandmark.INDEX_FINGER_TIP * 3 + 1]
    index_mcp_y = features[mp_hands.HandLandmark.INDEX_FINGER_MCP * 3 + 1]
    index_ip_y = features[mp_hands.HandLandmark.INDEX_FINGER_DIP * 3 + 1]
    # Middle
    middle_tip_y = features[mp_hands.HandLandmark.MIDDLE_FINGER_TIP * 3 + 1]
    middle_mcp_y = features[mp_hands.HandLandmark.MIDDLE_FINGER_MCP * 3 + 1]

    # Ring
    ring_tip_y = features[mp_hands.HandLandmark.RING_FINGER_TIP * 3 + 1]
    ring_mcp_y = features[mp_hands.HandLandmark.RING_FINGER_MCP * 3 + 1]

    # Pinky
    pinky_tip_y = features[mp_hands.HandLandmark.PINKY_TIP * 3 + 1]
    pinky_mcp_y = features[mp_hands.HandLandmark.PINKY_MCP * 3 + 1]

    # Example rules for simple gestures
    if (thumb_tip_y < thumb_ip_y < thumb_mcp_y and 
        thumb_tip_y < index_tip_y and thumb_tip_y < middle_tip_y and 
        thumb_tip_y < ring_tip_y and thumb_tip_y < pinky_tip_y and 
        index_mcp_y < middle_mcp_y and middle_mcp_y < ring_mcp_y and ring_mcp_y < pinky_mcp_y):
        return "Thumbs Up"
    elif (thumb_tip_y > thumb_ip_y > thumb_mcp_y and 
        thumb_tip_y > index_tip_y and thumb_tip_y > middle_tip_y and 
        thumb_tip_y > ring_tip_y and thumb_tip_y > pinky_tip_y and 
        index_mcp_y > middle_mcp_y and middle_mcp_y > ring_mcp_y and ring_mcp_y > pinky_mcp_y):
        return "Thumbs Down"
    elif (index_tip_y > middle_tip_y and middle_tip_y < ring_tip_y and 
        ring_tip_y < pinky_tip_y and 
        thumb_tip_y < thumb_ip_y and thumb_ip_y < thumb_mcp_y and pinky_mcp_y >pinky_tip_y and ring_mcp_y>ring_tip_y):
        return "Open Hand"
    elif (index_tip_y < index_mcp_y and middle_tip_y > middle_mcp_y and 
        ring_tip_y > ring_mcp_y and 
        index_tip_y <middle_tip_y and index_tip_y<thumb_tip_y):
        return "Pointing"
    elif (index_tip_y < index_mcp_y and middle_tip_y < middle_mcp_y and 
        ring_tip_y > ring_mcp_y and index_tip_y < index_ip_y and index_tip_y < thumb_tip_y and middle_tip_y < thumb_tip_y
        ):
        return "Peace Sign"
    if ( thumb_ip_y < thumb_mcp_y and thumb_tip_y < thumb_mcp_y and
        index_tip_y > index_mcp_y and middle_tip_y > middle_mcp_y and ring_tip_y > ring_mcp_y and
        pinky_tip_y >pinky_mcp_y):
        return "Fist"
    else:
        return "Unknown"




# Create your views here.






def start_page(request):
    global cap1,cap2
    if cap1!="":
        cap1.release()
    if cap2!="":
        cap2.release()
    return render(request,'start.html')




def gen_frames(request,todo):
    global cap1,name,cap2,gesture,ges_count,gestures_list,sno,to_verify,id_list,name_list
    # cap1 = cv2.VideoCapture(0)  
    if todo == 1:
        while True:
            success, frame = cap1.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            face_locations = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            for (x, y, w, h) in face_locations:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)


            if not success:
                break
            else:
                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')



    elif todo == 2:
        tobreak = False
        frame_count = 0
        while True:
            frame_count += 1
            success, frame = cap2.read()
            if frame_count>10 and frame_count%5==0:

                embedding = DeepFace.represent(frame, model_name='Facenet', enforce_detection=False)


                for i in range(len(to_verify)):
                    result = DeepFace.verify(embedding[0]['embedding'], to_verify[i],distance_metric="cosine",model_name='Facenet', enforce_detection=False, threshold=0.3)
                    print(result)
                    if result['verified']:
                        name = name_list[i//3]
                        print(name)
                        password = "ABCabc/--03mbns"
                        sno = id_list[i//3]
                        user = auth.authenticate(username=id_list[i//3],password=password)
                        print(user)
                        
                        if user is not None:
                            login(request,user)
                            print("login")
                            tobreak = True
                            break
            if tobreak:
                break 


            if not success:
                break
            else:
                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                
        
        gesture = rd.choice(gestures_list)
        print(gesture)
        if gesture == 'Mouth Open':
            print("inside")
            isgo = True
            while True:
                success, frame = cap2.read()
                landmarks = get_landmarks(frame)
                if landmarks is not None:
                    mouth_outer = landmarks[48:60]
                    mouth_dist = np.linalg.norm(mouth_outer[3] - mouth_outer[9])  # Vertical distance

                    # Define thresholds for mouth states
                    mouth_open_threshold = 30
                    mouth_closed_threshold = 25

                    if mouth_dist > mouth_open_threshold and isgo:  # Threshold for open mouth
                        cv2.putText(frame, "Mouth Open", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        ges_count -= 1
                        isgo = False
                        if ges_count == 0:
                            break
                    elif mouth_dist < mouth_closed_threshold:  # Threshold for closed mouth
                        cv2.putText(frame, "Mouth Closed", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        isgo = True

                if not success:
                    break
                else:
                    ret, buffer = cv2.imencode('.jpg', frame)
                    frame = buffer.tobytes()
                    yield (b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')



        elif gesture == 'Fist':
            isgo = True
            whlbk = False
            ct = 0
            while True:
                success, frame = cap2.read()

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                result = hands.process(rgb_frame)
                ver = True 
                if result.multi_hand_landmarks:
                    for hand_landmarks in result.multi_hand_landmarks:
                        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                        features = extract_features(hand_landmarks)
                        gest = recognize_gesture(features)
                        print(gest)
                        if gest=='Fist' and isgo and ct<=0:
                            cv2.putText(frame, gest, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                            
                            ges_count -=1
                            print(ges_count)
                            isgo = False
                            ct =5
                            if ges_count == 0:
                                ver =False
                                break 
                        elif gest!="No Gesture" and gest!= "Fist" :
                            ct -= 1
                            if ct<=0:
                                isgo =True
                                ct = 0
                            cv2.putText(frame, gest, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0),2,cv2.LINE_AA)
                    if not ver:
                            break



                if not success:
                    break
                else:
                    ret, buffer = cv2.imencode('.jpg', frame)
                    frame = buffer.tobytes()
                    yield (b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')        





        elif gesture == 'Thumbs Up':
            isgo = True
            cnt =0
            while True:
                success, frame = cap2.read()
     

# Define gesture recognition function

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame to find hands
                result = hands.process(rgb_frame)
                ver =True  
                if result.multi_hand_landmarks:
                        for hand_landmarks in result.multi_hand_landmarks:
                            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                            features = extract_features(hand_landmarks)
                            gest = recognize_gesture(features)
                        
                            if gest=='Thumbs Up' and isgo and cnt<=0:
                                cv2.putText(frame, gest, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                                ges_count -=1
                                print(ges_count)
                                isgo = False
                                cnt =5
                                if ges_count == 0:
                                    ver =False
                                    break 
                            elif gest!= 'No Gesture'  and gest != 'Thumbs Up':
                                cnt-=1
                                if  cnt<=0 :
                                    isgo =True
                                cv2.putText(frame, gest, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                        if not ver:
                            break

                    
                

                if not success:
                    break
                else:
                    ret, buffer = cv2.imencode('.jpg', frame)
                    frame = buffer.tobytes()
                    yield (b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        elif gesture == 'Peace Sign':
            isgo = True
            cnt =0
            while True:
                success, frame = cap2.read()
     

# Define gesture recognition function

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame to find hands
                result = hands.process(rgb_frame)
                ver =True   
                if result.multi_hand_landmarks:
                        for hand_landmarks in result.multi_hand_landmarks:
                            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                            features = extract_features(hand_landmarks)
                            gest = recognize_gesture(features)
                        
                            if gest=='Peace Sign' and isgo and cnt<=0:
                                cv2.putText(frame, gest, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                                ges_count -=1
                                print(ges_count)
                                isgo = False
                                cnt =5
                                if ges_count == 0:
                                    ver =False
                                    break 
                            elif gest!= 'No Gesture'  and gest != 'Peace Sign':
                                cnt-=1
                                if  cnt<=0 :
                                    isgo =True
                                cv2.putText(frame, gest, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                        if not ver:
                            break

                    

                if not success:
                    break
                else:
                    ret, buffer = cv2.imencode('.jpg', frame)
                    frame = buffer.tobytes()
                    yield (b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        elif gesture == 'Open Hand':
            isgo = True
            cnt =0
            while True:
                success, frame = cap2.read()
     

# Define gesture recognition function

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame to find hands
                result = hands.process(rgb_frame)
                ver = True    
                if result.multi_hand_landmarks:
                        for hand_landmarks in result.multi_hand_landmarks:
                            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                            features = extract_features(hand_landmarks)
                            gest = recognize_gesture(features)
                        
                            if gest=='Open Hand' and isgo and cnt<=0:
                                cv2.putText(frame, gest, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                                ges_count -=1
                                print(ges_count)
                                isgo = False
                                cnt =5
                                if ges_count == 0:
                                    ver=False
                                    break 
                            elif gest != 'No Gesture'  and gest != 'Open Hand':
                                cnt-=1
                                if  cnt<=0 :
                                    isgo =True
                                cv2.putText(frame, gest, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

                    
                
                        if not ver:
                            break

                if not success:
                    break
                else:
                    ret, buffer = cv2.imencode('.jpg', frame)
                    frame = buffer.tobytes()
                    yield (b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        elif gesture == 'Thumbs Down':
            isgo = True
            cnt =0
            while True:
                success, frame = cap2.read()
     

# Define gesture recognition function

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame to find hands
                result = hands.process(rgb_frame)
                ver = True   
                if result.multi_hand_landmarks:
                        for hand_landmarks in result.multi_hand_landmarks:
                            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                            features = extract_features(hand_landmarks)
                            gest = recognize_gesture(features)
                        
                            if gest=='Thumbs Down' and isgo and cnt<=0:
                                cv2.putText(frame, gest, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                                ges_count -=1
                                print(ges_count)
                                isgo = False
                                cnt =5
                                if ges_count == 0:
                                    ver =False
                                    break 
                            elif gest!= 'No Gesture'  and gest != 'Thumbs Down':
                                cnt-=1
                                if  cnt<=0 :
                                    isgo =True
                                cv2.putText(frame, gest, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                        if not ver:
                            break

                    
                

                if not success:
                    break
                else:
                    ret, buffer = cv2.imencode('.jpg', frame)
                    frame = buffer.tobytes()
                    yield (b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        elif gesture == 'Pointing':
            isgo = True
            cnt =0
            while True:
                success, frame = cap2.read()
     

# Define gesture recognition function

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame to find hands
                result = hands.process(rgb_frame)
                ver =True  
                if result.multi_hand_landmarks:
                        for hand_landmarks in result.multi_hand_landmarks:
                            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                            features = extract_features(hand_landmarks)
                            gest = recognize_gesture(features)
                        
                            if gest=='Pointing' and isgo and cnt<=0:
                                cv2.putText(frame, gest, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                                ges_count -=1
                                print(ges_count)
                                isgo = False
                                cnt =5
                                if ges_count == 0:
                                    ver =False
                                    break 
                            elif gest!= 'No Gesture'  and gest != 'Pointing':
                                cnt-=1
                                if  cnt<=0 :
                                    isgo =True
                                cv2.putText(frame, gest, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                        if not ver:
                            break

                    
                

                if not success:
                    break
                else:
                    ret, buffer = cv2.imencode('.jpg', frame)
                    frame = buffer.tobytes()
                    yield (b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')







def video_feed_1(request):
    return StreamingHttpResponse(gen_frames(request,1), content_type='multipart/x-mixed-replace; boundary=frame')



def video_feed_2(request):
    return StreamingHttpResponse(gen_frames(request,2), content_type='multipart/x-mixed-replace; boundary=frame')



def login_reactive_1(request):
    global name, gesture
    def name_live_detect():
            disp = ""
            if name != "":
                
                disp = "Face Detected for "+ str(name)
                print(disp)
            
            yield f"data: {disp}\n\n"
    
    return StreamingHttpResponse(name_live_detect(),content_type="text/event-stream")



def login_reactive_2(request):
    global name, gesture
    def name_live_detect():
        disp = ""             
        disp = "Gesture :"+str(gesture)    
        yield f"data: {disp}\n\n"

    return StreamingHttpResponse(name_live_detect(),content_type="text/event-stream")




def login_reactive_3(request):
    global ges_count
    def ges_live_detect():
        if ges_count==0:
            yield "data: redirect\n\n"
        else:
            disp = "Gesture Count: "+ str(ges_count)
            yield f"data: {disp}\n\n"
    
    return StreamingHttpResponse(ges_live_detect(),content_type="text/event-stream")



# def login_reactive_2(request):
#     global gesture
#     def ges_detect():
#         while True:
#             disp = ""
#             if gesture != "":
#                 disp = "Gesture : "+ str(gesture)
#             yield f"data: {disp}\n\n"
    
#     return StreamingHttpResponse(ges_detect(),content_type="text/event-stream")



# def login_reactive_3(request):
#     global ges_count
#     def ges_live_detect():
#         while True:
#             if ges_count==0:
#                 yield "data: redirect\n\n"
#                 break

#             disp = "Gesture Count: "+ str(ges_count)
#             yield f"data: {disp}\n\n"
    
#     return StreamingHttpResponse(ges_live_detect(),content_type="text/event-stream")




def register_count_stream(request):
    global recent_images
    def count_reg_face():
        
            reg_face_mess = "Face Count : "+str(len(recent_images))
            if len(recent_images)==10:
                reg_face_mess += str("\n Now You May Press the Register Button")
            yield f"data: {reg_face_mess}\n\n"
    
    return StreamingHttpResponse(count_reg_face(),content_type='text/event-stream')





def register(request):
    global reg_embeds,cap1
    cap1 = cv2.VideoCapture(0)
    if request.method == "POST":
        best_reg_embeds = sortemb(reg_embeds)
        print(best_reg_embeds)
        print(request.POST.get('username'))
        username = request.POST.get('username')
        password = "ABCabc/--03mbns"
        service = request.POST.get('sno')
        rank = request.POST.get('rank')
        prereg = Profile.objects.all()
        for i in prereg:
            if i.service_no==service:
                messages.success(request, "Not Registered : Same Service Number")
                return redirect('/')

        data = User.objects.create_user(username=service, password=password)


        proreg = Profile.objects.create(name=username, rank=rank,service_no=service)

        emb_reg_data = Embeds.objects.create(name=username,id=service,embed1=best_reg_embeds[0],embed2=best_reg_embeds[1],embed3=best_reg_embeds[2])
        messages.success(request, "Succesfully Registered")

        return redirect('/')

    return render(request,'register.html')





def start_camera_1(request):
    global cap1
    cap1 = cv2.VideoCapture(0)
    return redirect('register')








@csrf_exempt
def capture_image(request):
    global cap1, recent_images,reg_embeds
    frame_count, capture_count = 0, 0
    recent_images.clear()
    
    if request.method == 'POST':
        try:
            while True:
                success, frame = cap1.read()
                if not success:
                    return JsonResponse({'error': 'Failed to capture image'}, status=500)
                
                frame_count += 1
                print(f"Frame Count: {frame_count}, Capture Count: {capture_count}")
                
                if frame_count % 10 == 0:
                    embeddings = DeepFace.represent(frame, model_name='Facenet', enforce_detection=False)
                    print(embeddings)
                    reg_embeds.append(embeddings)
                    
                    _, buffer = cv2.imencode('.jpg', frame)
                    encoded_image = base64.b64encode(buffer).decode('utf-8')
                    
                    recent_images.append(encoded_image)
                    capture_count += 1
                    
                    if capture_count == 10:
                        break

            return JsonResponse({'recent_images': recent_images}, status=200)

        except Exception as e:
            print(e)
            return JsonResponse({'error': 'Internal Server Error'}, status=500)
    else:
        return JsonResponse({'error': 'Invalid request method'}, status=405)




def loginuser(request):
    global name, ges_count, cap2,to_verify,id_list,name_list,sno
    password = "ABCabc/--03mbns"
    # password=5
    user = auth.authenticate(username=sno,password=password)
    if user is not None:
        return redirect('/homepage/')
    cap2 = cv2.VideoCapture(0)
    name = ""  
    ges_count = 4
    embed_reg_data = Embeds.objects.all()
    tobreak = False
    for emb in embed_reg_data:
        emt1 = emb.embed1.split(',')
        emt1[0] = emt1[0][1:]
        emt1[-1] = emt1[-1][:-1]
        emt1 = list(map(float,emt1))
        to_verify.append(emt1)
        emt2 = emb.embed2.split(',')
        emt2[0] = emt2[0][1:]
        emt2[-1] = emt2[-1][:-1]
        emt2 = list(map(float,emt2))
        to_verify.append(emt2)
        emt3 = emb.embed3.split(',')
        emt3[0] = emt3[0][1:]
        emt3[-1] = emt3[-1][:-1]
        emt3 = list(map(float,emt3))
        to_verify.append(emt3)
        id_list.append(emb.id)
        name_list.append(emb.name)
    return render(request,'login.html')


def sortemb(emb):
    ca = []

    for i in range(len(emb)):
        ca.append([emb[i][0]['face_confidence'],i])
    
    ca.sort(reverse=True)
    print(ca)
    newemb = [emb[ca[0][-1]][0]['embedding'], emb[ca[1][-1]][0]['embedding'], emb[ca[2][-1]][0]['embedding']]
    return newemb



# def loginuser(request):
#     global cap2,ges_count,name
#     name = ""
#     ges_count = 4
#     cap2 = cv2.VideoCapture(0)
#     return render(request,'login.html')



def homepage(request):
    # global cap2,name,sno
    # password = "ABCabc/--03mbns"
    sno=5
    password=5
    user = auth.authenticate(username=sno,password=password)
    print(user)
    if user is not None:
        login(request,user)
        print("login")
    if cap2!="":
        cap2.release()
    if request.user.is_anonymous:
        return redirect('/')
    user_data = request.user
    return redirect(f'/myapps/?name={name}')


def logoutuser(request):
    # global name,sno
    sno = 5
    name = ""
    logout(request)
    return redirect('/')