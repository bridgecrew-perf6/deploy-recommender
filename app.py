import streamlit as st
import pickle
import pandas as pd
import numpy as np
from numpy import dot
from numpy.linalg import norm


# print functions

def fetchPoster(movie_name):
    movie_index=movies_list[movies_list['title_x']==movie_name]
    movie_poster=movie_index['posterpath'].values
    poster_url='https://image.tmdb.org/t/p/w200{}'.format("".join(movie_poster))
    return poster_url

def showDesc(movieName):
    movie_index=movies_list[movies_list['title_x']==movieName]
    st.markdown('_Genres:_ ')
    for genre in movie_index['genres'].values:
        st.text(genre)
    st.markdown('_Cast:_ ')
    for cast in movie_index['cast'].values:
        st.text(cast)
    

def show(df):
    for name in df.head(10).movie_name.values:
        if(fetchPoster(name)):
          col1, col2 = st.columns([1,3])

          with col1:
            st.image(fetchPoster(name))

          with col2:
            st.subheader(name)
            for percent in df[df['movie_name']==name]['finaldistance']:
                percent=percent*100
                percent=format(percent,".4f")
                st.subheader("{} % recommended".format(percent))
          
        else: 
            {}

def castGenrePrint(pop):
          col1, col2 = st.columns([1,3])

          with col1:
            st.image(fetchPoster(pop))

          with col2:
            st.subheader(pop)
            showDesc(pop)


# distance functions

def keywordDistances(userVector):
    distances=[]
    
    i=0
    for moviekeywordbin in movies_list['keyword_bin']:
        movie_id=movies_list.iloc[i].id
        movie_name=movies_list.iloc[i].title_x
        distance=trial(userVector,moviekeywordbin)
        distances.append((distance,movie_id,movie_name))
        i+=1
    return distances

def castDistances(userVector):
    distances=[]
    
    i=0
    for moviecastbin in movies_list['cast_bin']:
        movie_id=movies_list.iloc[i].id
        movie_name=movies_list.iloc[i].title_x
        distance=trial(userVector,moviecastbin)
        distances.append((distance,movie_id,movie_name))
        i+=1
    return distances

def directorDistances(userVector):
    distances=[]
    
    i=0
    for moviedirectorbin in movies_list['director_bin']:
        movie_id=movies_list.iloc[i].id
        movie_name=movies_list.iloc[i].title_x
        distance=trial(userVector,moviedirectorbin)
        distances.append((distance,movie_id,movie_name))
        i+=1
    return distances

def genreDistances(userVector):
    distances=[]
    
    i=0
    for moviegenrebin in movies_list['genre_bin']:
        movie_id=movies_list.iloc[i].id
        movie_name=movies_list.iloc[i].title_x
        distance=trial(userVector,moviegenrebin)
        distances.append((distance,movie_id,movie_name))
        i+=1
    return distances

def releasedEraDistances(userVector):
    distances=[]
    
    i=0
    for movieerabin in movies_list['released_bin']:
        movie_id=movies_list.iloc[i].id
        movie_name=movies_list.iloc[i].title_x
        distance=trial(userVector,movieerabin)
        distances.append((distance,movie_id,movie_name))
        i+=1
    return distances

def runtimeDistances(userVector):
    distances=[]
    
    i=0
    for movieRuntimebin in movies_list['runtime_bin']:
        movie_id=movies_list.iloc[i].id
        movie_name=movies_list.iloc[i].title_x
        distance=trial(userVector,movieRuntimebin)
        distances.append((distance,movie_id,movie_name))
        i+=1
    return distances


# helper functions

def LikedMovie(movieName,flag):
    userDF=st.session_state.df
    userDF=updateUserDF(userDF,movieName,flag)
    st.session_state.df=userDF 

def trial(a,b):
    
    if norm(a) != 0.0 and norm(b) != 0.0:
     cos_sim = dot(a, b)/(norm(a)*norm(b))
    else: 
     cos_sim=float(0)
    return cos_sim

def weightage(listname,weight):
    columns = list(zip(*listname))
    column=[x * weight for x in columns[0]] 
    
    listname=np.asanyarray(listname)
    column=np.array(column)

    listname[:,0] = column
    listname=list(listname)
    return listname

def addVectors(list1,list2):
    list1 = [float(x) + float(y) for (x, y) in zip(list1, list2)] 

    return list1

def subtractVectors(list1,list2):
    list1 = [float(x) - float(y) for (x, y) in zip(list1, list2)] 

    return list1

def check(list1,list2):
    list3=[]
    for (x,y) in zip(list1,list2):
        if(float(x)<=0.0 and y==1):
            list3.append(float(y))
        else:
            list3.append(float(x))
    return list3


# Main Function

def updateUserDF(userDF,movie_name,flag):
    movie_index=movies_list[movies_list['title_x']==movie_name]

    genreVector=movie_index['genre_bin'].values
    genreVector=genreVector.tolist()
    genreVector = [j for sub in genreVector for j in sub]
    
    castVector=movie_index['cast_bin'].values
    castVector=castVector.tolist()
    castVector = [j for sub in castVector for j in sub]
       
    directorVector=movie_index['director_bin'].values
    directorVector=directorVector.tolist()
    directorVector = [j for sub in directorVector for j in sub]

    keywordVector=movie_index['keyword_bin'].values
    keywordVector=keywordVector.tolist()
    keywordVector = [j for sub in keywordVector for j in sub]

    runtimeVector=movie_index['runtime_bin'].values
    runtimeVector=runtimeVector.tolist()
    runtimeVector = [j for sub in runtimeVector for j in sub]
       
    releasedVector=movie_index['released_bin'].values
    releasedVector=releasedVector.tolist()
    releasedVector = [j for sub in releasedVector for j in sub]

    if(flag==True):
     userDF.iloc[0].userGenreVector=addVectors(userDF.iloc[0].userGenreVector,genreVector)
   
     userDF.iloc[0].userCastVector=addVectors(userDF.iloc[0].userCastVector,castVector)

     userDF.iloc[0].userDirectorVector=addVectors(userDF.iloc[0].userDirectorVector,directorVector)

     userDF.iloc[0].userKeywordVector=addVectors(userDF.iloc[0].userKeywordVector,keywordVector)

     userDF.iloc[0].userRuntimeVector=addVectors(userDF.iloc[0].userRuntimeVector,runtimeVector)

     userDF.iloc[0].userReleasedEraVector=addVectors(userDF.iloc[0].userReleasedEraVector,releasedVector)
    else:
     userDF.iloc[0].userGenreVector=subtractVectors(userDF.iloc[0].userGenreVector,genreVector)
   
     userDF.iloc[0].userCastVector=subtractVectors(userDF.iloc[0].userCastVector,castVector)

     userDF.iloc[0].userDirectorVector=subtractVectors(userDF.iloc[0].userDirectorVector,directorVector)

     userDF.iloc[0].userKeywordVector=subtractVectors(userDF.iloc[0].userKeywordVector,keywordVector)

     userDF.iloc[0].userRuntimeVector=subtractVectors(userDF.iloc[0].userRuntimeVector,runtimeVector)

     userDF.iloc[0].userReleasedEraVector=subtractVectors(userDF.iloc[0].userReleasedEraVector,releasedVector)
    return userDF


def finalScorePredictor(userDF):
    
    gdistances=weightage(genreDistances(userDF.iloc[0].userGenreVector),0.3)
    kdistances=weightage(keywordDistances(userDF.iloc[0].userKeywordVector),0.2)
    cdistances=weightage(castDistances(userDF.iloc[0].userCastVector),0.2)
    redistances=weightage(releasedEraDistances(userDF.iloc[0].userReleasedEraVector),0.15)
    ddistances=weightage(directorDistances(userDF.iloc[0].userDirectorVector),0.1)
    rdistances=weightage(runtimeDistances(userDF.iloc[0].userRuntimeVector),0.05)
    
    c1 = list(zip(*gdistances))
    c2 = list(zip(*kdistances))
    c3=list(zip(*cdistances))
    c4=list(zip(*redistances))
    c5=list(zip(*ddistances))
    c6=list(zip(*rdistances))
    finaldistances=addVectors(c1[0],c2[0])
    finaldistances=addVectors(finaldistances,c3[0])
    finaldistances=addVectors(finaldistances,c4[0])
    finaldistances=addVectors(finaldistances,c5[0])
    finaldistances=addVectors(finaldistances,c6[0])
    data = {'finaldistance':finaldistances,
        'movie_name':movies_list_names ,
        }
    data_df=pd.DataFrame(data)
    data_df=data_df.sort_values(by='finaldistance',ascending=False)

    return data_df



# loading dataframes

movies_list=pickle.load(open('finalMovies.pkl','rb'))
movies_list_names=movies_list['title_x'].values
popularMoviesList=pickle.load(open('popularMovies.pkl','rb'))


# list of attribute

genreList=pickle.load(open('genreList.pkl','rb'))


# Session State

userDF=pickle.load(open('userDF.pkl','rb'))
if 'df' not in st.session_state:
    st.session_state.df = userDF


# Main page

st.title('Movie Recommender System')

placeholder = st.empty()
if not st.checkbox("Skip"):
    options = st.multiselect(
     'Choose your favorite genres',
     genreList,
     )

    genreUserVector=[]

    for genre in genreList:
     if genre in options:
        genreUserVector.append(1)
     else: 
        genreUserVector.append(0)

    if st.button('Submit'):
      
      userDF=st.session_state.df
      userDF.iloc[0].userGenreVector = check(userDF.iloc[0].userGenreVector, genreUserVector)
      st.session_state.df=userDF
    
      
selected_movie_name=st.selectbox('Enter your fav movie',movies_list_names)


if st.button('Search'):

    castGenrePrint(selected_movie_name)
    LikedMovie(selected_movie_name,True)
    st.text("")
    st.text("")

st.text("")
st.text("")
if st.button('Recommend Movies!'):
    
    userDF=st.session_state.df
    df=finalScorePredictor(userDF)
    st.subheader("Movies according to your taste!")
    show(df)


st.header("Popular movies you may like!")
          
for pop in popularMoviesList['title_x'].head(10):

    if(fetchPoster(pop)):
          castGenrePrint(pop)
          if st.button('Like',key=pop):
            LikedMovie(pop,True)
     
          if st.button('Dislike',key=pop+"1"):
            LikedMovie(pop,False)
    else: 
            {}
    st.text("")

