# consulting

attention --> 트윗 해시태그 다는 코드 & 그걸 응용한 키워드 추천하는 코드

cnn_topic_classification --> cnn으로 토픽분류하는 코드

parse --> 키워드 추출 관련 코드 (backup)

keyword --> 키워드 추출 관련 코드 (latest)

https://www.slideshare.net/mobile/taejoonyoo/voc-20151030


전제 : 1 member 는 

 1현 프로세스의 모듈화 / 자동화 / 속도 향상
    - 모듈화 : 각 function을 분리 할 수 있도록 조정 -> 타인이 function call 만으로 전체 프로세스를 조정할 수 있도록 수정
    - 자동화 : 전처리 -> 분석 -> 결과 과정을 자동으로 진행하여, 사용자가 두,세번의 command로 원하는 결과물 추출
    - 속도 향상 : 전처리, 분석, 결과 추출 과정 중 병목 현상이 일어나는 부문을 일부 cpu 병렬처리를 통해 속도를 향상시키도록 수정
   


@전처리 단계 
전체 일정 중 절반 이상을 소요 할 것으로 예상
(6주)

- tokenizing
    - 재사용 예정
- 불용어 처리
    - 유의어 사전에 대한 지원이 필요
    - 업무 전문가를 초빙하여 의미적으로 빠져야 하는 것들을 지원을 받도록 함
    - STT 불순물 정제를 위한 불용어 리스트 필요
- 감성 감정 불만 사전에 작업
    - 오토마타 욕설 관련 알고리즘 활용 (1 member)
- 동사, 형용사 추가 stemming / 정제
    - 넥스투비 주도로 작업

@Word Vector / Word Network 구성
(4주 ~ 8주)

요청사항 : hardware GPU 필요

	- Doc or Sentence 별로 skip gram 기반 vector  & n-gram 기반 vector 추가
		- 지속적 논의
		- 복합명사 & clustering 적용 가능
	- Word Network
		- 재사용 예정
		- business적으로 의미가 있다고 생각되는 리스트에 맞춰서 가중치 조정 (parameter tuning)
		- graph package 설치 필요 예상

(6주 ~ 10주)
@ paragraph 분류
	- 기존 word network 기반 keyword 로 분류
	- Sentence Vector 추가해서 의미별 분류 시도
		- paragraph 별로 주제에 대한 label 요청
		- paragraph의 분리 validation 요청

(6주 ~ 10주)
@ keyword 묶음 고도화
- keyword 묶음 프레이즈 화
- business 의미적으로 묶일수 있는 알고리즘 시도


(8주~ 16주)
@ 결과물 추출
1. clustering(비지도) / 
	- k-nn / SVM / PCA  와 같은 기법 시도
	- keyword 묶음 기반으로 clustering 시도
   2. topic classification(지도)
      - call 단위로 label 데이터 요청
   3. 고객의 가입문의 / 불만 추출
      - 가입문의 / 불만의 강도와 관련된 call 또는 단어의 label 데이터 요청
     - 
 


      - 10 개 중 1개가 keyword 묶음에서 추출되었다


향후 / 예외) 고객 legacy 결합 하여 의미적 결과물 추출 

기타 요청사항 ) 모듈 설치 권한 요청

role
이태희 수석님 : 업무조율, 동사, 형용사 추가 stemming / 정제, 개발 표준 정의, Output & UI DB 연결

정윤수 차장님 : performance, 소스코드 inspection 세션 일정 및 세부 관리,  

곽태민 매니저 : clustering / 불만 오토마타 적용 / sentence skip-gram (n-gram) 적용
여진수 매니저 : performance / 모듈화 / function화 / parallel 
000 매니저 : 키워드 묶음 / clustering / 불만 내역 —> 데이터 검증 parameter tuning 
000 매니저 : 불만 오토마타 / 가입문의 / 일반 세부 내역 추출 
000 매니저 : paragraph 의미별 추출 / 분리
000 매니저 : 키워드 묶음 프레이즈화


지원 :
사용자 지정사전 관리
유의어 사전 / 불용어 사전 추가
paragraph validation / 주제 label 화
topic classification(지도) : call 단위로 label 데이터 요청
가입문의/불만 강도와 관련된 call 또는 단어의 label 데이터 요청
