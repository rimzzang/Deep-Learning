# Deep-Learning
1. coco data를 AI challenge 형태로 변환하기

head_point = cv2.circle(img_resize,
                            ((set_p[2][0]+set_p[1][0])//2,
                             (set_p[2][1]+set_p[1][1])*3//2-set_p[0][1]*2)
                            , 5, (255, 0, 0), cv2.FILLED)
    neck_point = cv2.circle(img_resize,
                            ((set_p[6][0]+set_p[5][0])//2,
                             (set_p[6][1]+set_p[5][1])//3+ set_p[0][1]//3)
                            , 5, (255, 0, 0), cv2.FILLED)
                            
coco dataset : { 코, 왼쪽 눈, 오른쪽 눈, 왼쪽 귀, 오른쪽 귀, 왼쪽 어깨, 오른쪽 어깨, 왼쪽 팔꿈치, 오른쪽 팔꿈치, 왼쪽 손목, 오른쪽 손목,
                왼쪽 골반, 오른쪽 골반, 왼쪽 무릎, 오른쪽 무릎, 왼쪽 발목, 오른쪽 발목 }

AI challenge : { 머리, 목, 왼쪽 어깨, 오른쪽 어깨, 왼쪽 팔꿈치, 오른쪽 팔꿈치, 왼쪽 손목, 오른쪽 손목,
                왼쪽 골반, 오른쪽 골반, 왼쪽 무릎, 오른쪽 무릎, 왼쪽 발목, 오른쪽 발목 }
                
                
#GOAL : coco data에서 코, 양쪽 눈, 양쪽 귀를 제외하고 기존의 점 위치를 활용해 머리와 목을 표현해야함

##i can do it
