
#load the package
library(tidyverse)

#Load the data into the workspace
path <- file.path('C:\\Users\\mehul\\OneDrive\\Desktop\\titanic_new_data\\final_train.csv')
data <- read_csv(path)

#Anaylsing the data
head(data) #overview of how data looks
tail(data)
summary(data) #summary points to know mean,median, Na values/missing values and more points
glimpse(data) #structure of the data - to check character, number class variables

#surival rate on the basis of Sex
data %>% ggplot(aes(x = Sex,y = Survived)) + geom_col(aes(fill = Sex))
#observation females survial rate was much more than males

#Pclass corelation with survivours
pie_data <- data %>% group_by(Pclass) %>% summarise(Survival_rate = mean(Survived))
pie(x = pie_data$Survival_rate, labels = pie_data$Pclass)
#Class1 people had more chance of survival, followed by Class2 and then Class3

#Age gap of people that traveled 
data %>% ggplot(aes(Age)) + geom_bar(fill = "#00bfff") #we can see maximum number of people from age gap of 20-40 travelled

#survival rate on the basis of Ageband
data %>% mutate(AgeBand = case_when(Age<=20 ~ '0-20',Age>20 & Age <=40 ~ '21-40', Age>40 & Age <= 60 ~ '41-60',Age > 60 ~ '61-80')) %>% group_by(AgeBand) %>% summarise(Survival_rate = mean(Survived))
#0-20 being children have higher tendency of survival
#some missing values too, to be taken care of

#load test data
testpath <- file.path('C:\\Users\\mehul\\OneDrive\\Desktop\\titanic_new_data\\final_test.csv')
testData <- read_csv(testpath)

#Starting data preparation - moving 'survived' column to the right end
data <- data %>% select(PassengerId,Pclass:Embarked,Survived)

#combining the data to perform same operations on both train and test data
combineData <- rbind(data[1:11],testData)

#taking care of missing values - Age
combineData$Age <- if_else(is.na(combineData$Age),round(mean(combineData$Age, na.rm = TRUE),2),combineData$Age) # replacing the missing values with their mean
sum(is.na(combineData$Age)) #recheck if there are any missing values

#taking care of missing values - Age
combineData$Fare <- if_else(is.na(combineData$Fare),round(mean(combineData$Fare, na.rm = TRUE),2),combineData$Fare) # replacing the missing values with their mean
sum(is.na(combineData$Fare)) #recheck if there are any missing values

#Creating AgeBand
combineData <- combineData %>% mutate(AgeBand = case_when(Age<=20 ~ '0-20',Age>20 & Age <=40 ~ '21-40', Age>40 & Age <= 60 ~ '41-60',Age > 60 ~ '61-Max'))
head(combineData,10)
#Spreading AgeBand into features
combineData <- combineData %>% spread(key = AgeBand,value = AgeBand) #values will be represented same as the feature name  
head(combineData,10)

#On the basis of Sibling/spouse and parent/child we can come up with a combine feature Family Members
data %>% mutate(FamilyM = SibSp+Parch) %>% group_by(FamilyM) %>% summarise(Survival_Rate = mean(Survived)) #viewing the probability of survival

#we can also be observe that number of people that travelled alone were very high
combineData %>% mutate(isAlone = if_else(SibSp+Parch == 0,1,0)) %>% count(isAlone) 

#both of the features can play vital role in building model, so adding these new features in dataset
combineData <- combineData %>% mutate(isAlone = if_else(SibSp+Parch == 0,1,0)) 
combineData <- combineData %>% mutate(FamilyM = SibSp+Parch)
combineData <- combineData %>% mutate(Familyband = case_when(FamilyM<=3 ~ '0-3',FamilyM>3 ~ '4-Max')) #added FamilyBand

#spreading the Band Values
combineData <- combineData %>% spread(key = Familyband,value = Familyband)

#Few feaures that will be removed from the dataset
combineData <- combineData %>% select(-Name,-SibSp,-Parch,-Cabin,-Ticket) #Name and ticket donot make sense for survival rate, Cabin has lots of null value and SibSp/Parch are already considered as joined FamilyM feature
#we will not be including them in are model
head(combineData)

#Analysing survival rate on the basis of Embarkation
data %>% group_by(Embarked) %>% summarise(Survival_Rate = mean(Survived)) #found some na values, will be dealt with

#max Embarked port
combineData %>% count(Embarked) %>% arrange(desc(n)) # as shown 'S' is the mode value for the feature

#replacing missing value of Embarked
combineData$Embarked <- if_else(condition = is.na(combineData$Embarked),'S',combineData$Embarked)

#spreading the feature Embarked
combineData <- spread(combineData,key = Embarked,value = Embarked)
head(combineData)

#Analysing Fare values with survival rate
data %>% ggplot(aes(x = Fare)) + geom_histogram(aes(fill = Survived)) + facet_grid(~Survived) #many people with less fare value
#less fare value less survival rate too

#creating FareBand
combineData <- combineData %>% mutate(Fareband = case_when(Fare<=32 ~ '0-32',Fare>32&Fare<=100 ~ '32-100',Fare>100 ~ '100-Max'))

#spreading FareBand
combineData <- combineData %>% spread(key = Fareband,value = Fareband)
head(combineData)

#spreading Pclass
combineData <- combineData %>% spread(key = Pclass,value = Pclass)
head(combineData)

#Changing value for 'Sex' feature to [0,1]
combineData <- combineData %>% mutate(Sex = if_else(Sex=='male',1,0))
head(combineData)

#changing vaues for other created features to [0,1]
nf <- c(5:8,11:21)
for(i in nf){
    combineData[i] <- if_else(is.na(combineData[i]),0,1)
}

head(combineData)

#Feature Scaling for Age and Fare as there values are very high and can put extra weight to the prediction
combineData <- combineData %>% mutate(Age = (Age - min(Age))/diff(range(Age)))
combineData <- combineData %>% mutate(Fare = (Fare - min(Fare))/diff(range(Fare)))
combineData <- combineData %>% mutate(FamilyM = (FamilyM - min(FamilyM))/diff(range(FamilyM)))
#performed Max-Min normalization not mean normalization, so that values can be in range [0,1]

#rounding the digits to 2 digit place, so that not to cause much noise in data
combineData$Age <- round(combineData$Age,2)
combineData$Fare <- round(combineData$Fare,2)

head(combineData)

#Data is all ready now, we can also remove PassengerId here
combineData <- combineData %>% select(-PassengerId)

#we also have to separate data to train and test
normalised_train_data <- cbind(combineData[1:891,],data[,"Survived"]) #first 891 rows and last Survived feature are part of training data
normalised_test_data <- combineData[892:1309,] #last 418 rows to test

#exporting the data
write.table(normalised_train_data,file = 'C:\\Users\\mehul\\OneDrive\\Desktop\\normalised_train_data.txt',sep = '\t', col.names = FALSE, row.names = FALSE)
write.table(normalised_test_data,file = 'C:\\Users\\mehul\\OneDrive\\Desktop\\normalised_test_data.txt',sep = '\t', col.names = FALSE, row.names = FALSE)


#Next steps for prediction are provided in Octave by me. Please have a look to it, to gain the accuracy of created model.
