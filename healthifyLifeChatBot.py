from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier, _tree
import numpy as np
from sklearn.model_selection import train_test_split
import warnings
from gensim.summarization import keywords
import pandas as pd
import csv
import os
import tkinter
from tkinter import *
from tkinter import messagebox
from PIL import ImageTk, Image

warnings.filterwarnings("ignore", category=DeprecationWarning)

training = pd.read_csv('Training.csv')
testing = pd.read_csv('Testing.csv')
cols = training.columns
cols = cols[:-1]
x = training[cols]
y = training['prognosis']
reduced_data = training.groupby(training['prognosis']).max()
le = preprocessing.LabelEncoder()
le.fit(y)
y = le.transform(y)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=42)
testx = testing[cols]
testy = testing['prognosis']
testy = le.transform(testy)

clf1 = DecisionTreeClassifier()
clf = clf1.fit(x_train, y_train)

severityDictionary = {}
description_list = {}
precautionDictionary = {}
symptoms_dict = {}
hospitals = {}
pincode = " "
doctors = pd.DataFrame()
doctors['name'] = np.nan
doctors['link'] = np.nan
doctors['disease'] = np.nan

for index, symptom in enumerate(x):
    symptoms_dict[symptom] = index


def getHospitals():
    global hospitals
    with open('hospital_directory.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if row[11] not in hospitals:
                _hospital = {row[11]: []}
                hospitals.update(_hospital)
            hospitals[row[11]].append(row[3] + "->" + row[2])


def calc_condition(exp, days):
    sumDays = 0
    for item in exp:
        sumDays += severityDictionary[item]
    if (sumDays * days) / (len(exp) + 1) > 13:
        printBot("You should take the consultation from doctor. ")
    else:
        printBot("It might not be that bad but you should take precautions and visit doctor if it persists..")


def getDescription():
    global description_list
    with open('symptom_Description.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            _description = {row[0]: row[1]}
            description_list.update(_description)


def getSeverityDict():
    global severityDictionary
    with open('symptom_severity.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        try:
            for row in csv_reader:
                _diction = {row[0]: int(row[1])}
                severityDictionary.update(_diction)
        except:
            pass


def getprecautionDict():
    global precautionDictionary
    with open('symptom_precaution.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            _prec = {row[0]: [row[1], row[2], row[3], row[4]]}
            precautionDictionary.update(_prec)


def getInfo():
    print("Your Name\t", end="->")
    name = input("")
    print("Hello ", name)
    pincode = input("Enter your Pincode:")


def getDoctors():
    global doctors
    doc_dataset = pd.read_csv('doctors_dataset.csv', names=['Name', 'Description'])
    diseases = reduced_data.index
    diseases = pd.DataFrame(diseases)
    doctors['disease'] = diseases['prognosis']
    # print(diseases)
    doctors['name'] = doc_dataset['Name']
    doctors['link'] = doc_dataset['Description']
    # print(doctors)


def printBot(output):
    # Diagnosis.selfRef.botMessage.delete(0.0, END)
    Diagnosis.selfRef.botMessage.insert(END, str(output) + "\n")


def printBot2(output):
    Diagnosis.selfRef.botMessage.insert(END, str(output) + "\n")


def sec_predict(symptoms_exp):
    df = pd.read_csv('Training.csv')
    X = df.iloc[:, :-1]
    y = df['prognosis']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=20)
    rf_clf = DecisionTreeClassifier()
    rf_clf.fit(X_train, y_train)

    symptoms_dict = {}

    for index, symptom in enumerate(X):
        symptoms_dict[symptom] = index

    input_vector = np.zeros(len(symptoms_dict))
    for item in symptoms_exp:
        input_vector[[symptoms_dict[item]]] = 1

    return rf_clf.predict([input_vector])


def print_disease(node):
    node = node[0]
    val = node.nonzero()
    disease = le.inverse_transform(val[0])
    return disease


def FindDisease(text, header):
    wordList = text.split(" ")
    count = 0
    for word in wordList:
        count += 1
    keyWordsList = keywords(text, split=" ", words=count / 2)
    print(keyWordsList)
    finalList = []
    for word in keyWordsList:
        for w in word.split(" "):
            finalList.append(w)
    matchingSymptoms = set()
    for head in header:
        for symptoms in finalList:
            if symptoms in head:
                matchingSymptoms.add(head)
    regexp = re.compile(text)
    for item in header:
        if regexp.search(item):
            matchingSymptoms.add(item)
    print(matchingSymptoms)
    if len(matchingSymptoms) < 1:
        return 0, header[0]
    else:
        return 1, list(matchingSymptoms)


def PredictionTree(tree, feature_names):
    tree_ = tree.tree_
    print(tree_)
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    chk_dis = ",".join(feature_names).split(",")
    # print(chk_dis)
    symptoms_present = []

    # conf_inp=int()
    printBot("Mention the symptoms you are experiencing briefly for accurate diagnosis.\t")
    while True:
        disease_input = Diagnosis.GetAnswer(Diagnosis.selfRef)
        # conf, cnf_dis = check_pattern(chk_dis, disease_input)
        conf, cnf_dis = FindDisease(disease_input, chk_dis)
        if conf == 1:
            printBot("searches related to input: ")
            for num, it in enumerate(cnf_dis):
                printBot2(str(num) + "->" + it)
            if len(cnf_dis) != 1:
                printBot2(f"Select the one which is most apt in your case (0 - {len(cnf_dis) - 1}):  ")
                conf_inp = int(Diagnosis.GetAnswer(Diagnosis.selfRef))
                while conf_inp < 0 or conf_inp > len(cnf_dis) - 1:
                    printBot(f"Enter a valid input: (0 - {len(cnf_dis) - 1})\t")
                    conf_inp = int(Diagnosis.GetAnswer(Diagnosis.selfRef))
            else:
                conf_inp = 0
            disease_input = cnf_dis[conf_inp]
            break
        else:
            printBot("Please describe your symptoms more briefly. I was not able to understand.")
    printBot("Okay. From how many days have you been experiencing this? : ")
    while True:
        try:
            num_days = int(Diagnosis.GetAnswer(Diagnosis.selfRef))
            break
        except ValueError:
            print("Entered exception")
            printBot("Enter just the number of days.")
    nodesCovered = set()
    nodesCovered.add(disease_input)

    def recurse(node, depth):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            # print(name,threshold)
            printBot("Are you also experiencing " + name + "?\t")
            while True:
                inp = Diagnosis.GetAnswer(Diagnosis.selfRef)
                if inp == "yes" or inp == "no":
                    break
                else:
                    printBot2("provide proper answers i.e. (yes/no):")
            if inp == "yes":
                nodesCovered.add(name)
            if name in nodesCovered:
                # if name == disease_input:
                val = 1
            else:
                val = 0
            if val <= threshold:
                # print("came left")
                recurse(tree_.children_left[node], depth + 1)
            else:
                # print("went right")
                symptoms_present.append(name)
                recurse(tree_.children_right[node], depth + 1)
        else:
            present_disease = print_disease(tree_.value[node])
            print(" 1st prediction " + present_disease)
            red_cols = reduced_data.columns
            symptoms_given = red_cols[reduced_data.loc[present_disease].values[0].nonzero()]
            symptoms_exp = []
            for syms in list(symptoms_given):
                printBot("Are you also experiencing any " + syms + "? : ")
                while True:
                    inp = Diagnosis.GetAnswer(Diagnosis.selfRef)
                    if inp == "yes" or inp == "no":
                        break
                    else:
                        printBot2("provide proper answers i.e. (yes/no) : ")
                if inp == "yes":
                    symptoms_exp.append(syms)

            second_prediction = sec_predict(symptoms_exp)
            # print(second_prediction)
            calc_condition(symptoms_exp, num_days)
            if present_disease[0] == second_prediction[0]:
                printBot("You may have " + present_disease[0])
                printBot2(description_list[present_disease[0]])
            else:
                printBot("You may have " + present_disease[0] + "or " + second_prediction[0])
                printBot2(description_list[present_disease[0]])
                printBot2(description_list[second_prediction[0]])

            precaution_list = precautionDictionary[present_disease[0]]
            printBot2("Take following measures : ")
            for i, j in enumerate(precaution_list):
                printBot2(str(i + 1) + ")" + j)

            confidence_level = (1.0 * len(nodesCovered)) / len(symptoms_given)
            print("confidence level is " + str(confidence_level))

            row = doctors[doctors['disease'] == present_disease[0]]
            strData = 'Consult ' + str(row['name'].values)
            printBot2(strData)
            strData = 'Visit ' + str(row['link'].values[0])
            printBot2(strData)
            printBot2("Enter your Pincode:")
            pincode = Diagnosis.GetAnswer(Diagnosis.selfRef)

            if pincode in hospitals:
                printBot2("The following is list of hospitals nearby you you can visit: ")
                for hosp in hospitals[pincode]:
                    printBot2(hosp)
            else:
                printBot2("Sorry we were not able to find nearby hospitals for that pincode. Make sure the pincode is "
                          "correct or enter a nearby pincode.")

    recurse(0, 1)


def Close():
    window.destroy()


def ClearScreen(parent):
    if parent:
        for widget in parent.pack_slaves():
            widget.forget()


def ReturnToMain():
    ClearScreen(Login.Root)
    ClearScreen(Register.Root)
    ClearScreen(About.Root)
    MainScreen.pack()


def LoginClick():
    ClearScreen(MainForm.Root)
    LoginForm = Login(MainForm.Root)
    LoginForm.pack()


def RegisterClick():
    ClearScreen(MainForm.Root)
    RegisterForm = Register(MainForm.Root)
    RegisterForm.pack()


def AboutClick():
    ClearScreen(MainForm.Root)
    AboutScreen = About(MainForm.Root)
    AboutScreen.pack()


def LoadDiagnosis():
    ClearScreen(MainForm.Root)
    DiagnosisScreen = Diagnosis(MainForm.Root)
    DiagnosisScreen.pack()


def Start():
    Diagnosis.selfRef.botMessage.delete(0.0, END)
    Diagnosis.selfRef.userMessage.delete(0.0, END)
    print("Analysis Started")
    PredictionTree(clf, cols)
    print("Analysis Ended")


class Diagnosis(Frame):
    selfRef = None
    objIter = None

    def __init__(self, frame=None):
        Diagnosis.selfRef = self
        super().__init__(master=frame)
        frame.title("Your Diagnosis")
        frame.geometry("700x900")
        self["bg"] = "light blue"
        imgChatBot = Image.open("Doctor.png")
        imgChatBot = imgChatBot.resize((80, 80))
        self.imageChatBot = ImageTk.PhotoImage(imgChatBot)
        self.imageBot = Label(self, image=self.imageChatBot)
        imgUsers = Image.open("user.png")
        imgUsers = imgUsers.resize((80, 80))
        self.imageChatUser = ImageTk.PhotoImage(imgUsers)
        self.imageUser = Label(self, image=self.imageChatUser)
        self.scrollbar = Scrollbar(self.selfRef)
        self.botMessage = Text(self, bd=0, bg="black", height=30, width=60, yscrollcommand=self.scrollbar.set, fg='#fff')
        self.userMessage = Text(self, height=8, width=60)
        self.okVar = tkinter.IntVar()
        self.btnAnswer = Button(self, text="Answer", height="1", bg="#0066ff", fg="#ffffff",
                                command=lambda: self.okVar.set(1))
        self.btnStart = Button(self, text="Start / ReStart", height="2", width="50", bg="#0066ff", fg="#ffffff",
                               command=Start)
        self.btnBack = Button(self, text="Logout", height="2", width="50", bg="#0066ff", fg="#ffffff",
                              command=ReturnToMain)
        self.btnExit = Button(self, text="Exit", height="2", width="50", bg="#0066ff", fg="#ffffff",
                              command=Close)
        self.Header = Label(self, text="Your Medical Health Assistant", bg="#0047b3", fg="#ffffff", width="50",
                            height="4", font="bold")
        self.createWidgets()

    def createWidgets(self):
        self.Header.grid(row=1, column=1, columnspan=3, pady=5, sticky=W + E)
        self.imageBot.grid(row=2, column=1, pady=2)
        self.botMessage.grid(row=2, column=2, columnspan=2, pady=2)
        self.imageUser.grid(row=3, column=3, pady=2)
        self.userMessage.grid(row=3, column=1, columnspan=2, rowspan=2, pady=2)
        self.btnAnswer.grid(row=4, column=3, columnspan=1, pady=2)
        self.scrollbar.grid(row=2, column=4, sticky=N+S+W, rowspan=1)
        self.btnStart.grid(row=5, column=1, pady=3, padx=4, sticky=W + E, columnspan=3)
        self.btnBack.grid(row=6, column=1, pady=3, sticky=W + E, columnspan=3)
        self.btnExit.grid(row=7, column=1, pady=3, sticky=W + E, columnspan=3)

    def GetAnswer(self):
        print("waiting")
        self.btnAnswer.wait_variable(self.okVar)
        print("Resumed")
        UserInput = self.userMessage.get("1.0", END)
        Diagnosis.selfRef.botMessage.insert(END, "YOU: "+str(UserInput) + "\n")
        self.userMessage.delete(0.0, END)
        UserInput.strip()
        UserInput = UserInput[:-1]
        print(UserInput)
        return UserInput


class MainForm(Frame):
    Root = None

    def __init__(self, frame=None):
        MainForm.Root = frame  # Here Root is Window passed.
        super().__init__(master=frame, bd=0, bg="black")
        frame.geometry("500x350")
        frame.title("Welcome to HealthifyLife")
        # self["bg"] = "light blue"
        self.Header = Label(self, text="Your Medical Health Assistant", bg="#0047b3", fg="#ffffff", width="50",
                            height="4", font="bold")
        self.btnLogin = Button(self, text="Login", height="2", width="10", bg="#0066ff", fg="#ffffff",
                               font="bold", command=LoginClick)
        self.btnRegister = Button(self, text="Register", height="2", width="10", bg="#0066ff", fg="#ffffff",
                                  font="bold", command=RegisterClick)
        self.btnAbout = Button(self, text="About", height="2", width="10", bg="#0066ff", fg="#ffffff",
                               font="bold", command=AboutClick)
        self.btnExit = Button(self, text="Exit", height="2", width="10", bg="#0066ff", fg="#ffffff",
                              font="bold", command=Close)
        self.createWidgets()

    def createWidgets(self):
        self.Header.grid(row=1, pady=5, sticky=W + E)
        self.btnLogin.grid(row=2, pady=5, sticky=W + E)
        self.btnRegister.grid(row=3, pady=5, sticky=W + E)
        self.btnAbout.grid(row=4, pady=5, sticky=W + E)
        self.btnExit.grid(row=5, pady=5, sticky=W + E)


class Login(Frame):
    Root = None

    def __init__(self, frame=None):
        Login.Root = frame
        super().__init__(master=frame)
        frame.title("Login HealthifyLife")
        frame.geometry("500x350")
        self["bg"] = "light blue"
        self.Header = Label(self, text="Enter your Credentials:", bg="#0047b3", fg="#ffffff", height="2")
        self.UsernameText = Label(self, text="Username: ", font="bold")
        self.UsernameInput = StringVar()
        self.UsernameBox = Entry(self, textvariable=self.UsernameInput)
        self.passwordText = Label(self, text="Password: ", font="bold")
        self.passwordInput = StringVar()
        self.passwordBox = Entry(self, textvariable=self.passwordInput, show='*')
        self.btnLogin = Button(self, text="Login", height="2", width="10", bg="#0066ff", fg="#ffffff",
                               font="bold", command=self.LoginUser)
        self.btnBack = Button(self, text="Return to Main Screen", height="2", width="10", bg="#0066ff", fg="#ffffff",
                              command=ReturnToMain)
        self.btnExit = Button(self, text="Exit", height="2", width="10", bg="#0066ff", fg="#ffffff",
                              command=Close)
        self.createWidgets()

    def createWidgets(self):
        self.Header.grid(row=1, column=1, pady=5, sticky=W + E, columnspan=2)
        self.UsernameText.grid(row=2, column=1, pady=3)
        self.UsernameBox.grid(row=2, column=2, pady=3)
        self.passwordText.grid(row=3, column=1, pady=3)
        self.passwordBox.grid(row=3, column=2, pady=3)
        self.btnLogin.grid(row=4, column=1, pady=5, sticky=W + E, columnspan=2)
        self.btnBack.grid(row=5, column=1, pady=5, sticky=W + E, columnspan=2)
        self.btnExit.grid(row=6, column=1, pady=5, sticky=W + E, columnspan=2)

    def LoginUser(self):
        username = self.UsernameBox.get()
        password = self.passwordBox.get()
        files = os.listdir()
        if username in files:
            file = open(username, "r")
            content = file.readlines()
            if password == content[1]:
                messagebox.showinfo("Success", "Login Successful")
                LoadDiagnosis()
            else:
                messagebox.showinfo("Failure", "Wrong Password Entered")
        else:
            messagebox.showinfo("Failure", "No Such User found. Register Yourself")


class Register(Frame):
    Root = None

    def __init__(self, frame=None):
        Register.Root = frame
        super().__init__(master=frame)
        frame.title("Register HealthifyLife")
        frame.geometry("500x350")
        self["bg"] = "light blue"
        self.Header = Label(self, text="Create your Credentials", bg="#0047b3", fg="#ffffff", height="2")
        self.UsernameText = Label(self, text="Username: ", font="bold")
        self.UsernameInput = StringVar()
        self.UsernameBox = Entry(self, textvariable=self.UsernameInput)
        self.passwordText = Label(self, text="Password: ", font="bold")
        self.passwordInput = StringVar()
        self.passwordBox = Entry(self, textvariable=self.passwordInput, show='*')
        self.btnRegister = Button(self, text="Register", height="2", width="10", bg="#0066ff", fg="#ffffff",
                                  font="bold", command=self.registerUser)
        self.btnBack = Button(self, text="Return to Main Screen", height="2", width="10", bg="#0066ff", fg="#ffffff",
                              command=ReturnToMain)
        self.btnExit = Button(self, text="Exit", height="2", width="10", bg="#0066ff", fg="#ffffff",
                              command=Close)
        self.createWidgets()

    def createWidgets(self):
        self.Header.grid(row=1, column=1, pady=5, sticky=W + E, columnspan=2)
        self.UsernameText.grid(row=2, column=1, pady=3)
        self.UsernameBox.grid(row=2, column=2, pady=3)
        self.passwordText.grid(row=3, column=1, pady=3)
        self.passwordBox.grid(row=3, column=2, pady=3)
        self.btnRegister.grid(row=4, column=1, pady=5, sticky=W + E, columnspan=2)
        self.btnBack.grid(row=5, column=1, pady=5, sticky=W + E, columnspan=2)
        self.btnExit.grid(row=6, column=1, pady=5, sticky=W + E, columnspan=2)

    def registerUser(self):
        file = open(self.UsernameBox.get(), "w")
        file.write(self.UsernameBox.get() + "\n")
        file.write(self.passwordBox.get())
        file.close()
        messagebox.showinfo("Success", "Registration Successful")
        ReturnToMain()


class About(Frame):
    Root = None

    def __init__(self, frame=None):
        About.Root = frame
        super().__init__(master=frame)
        frame.title("About HealthifyLife")
        frame.geometry("500x350")
        self["bg"] = "light blue"
        self.Header = Label(self, text="About HealthifyLife", bg="#0047b3", fg="#ffffff", height="2")
        self.information = Text(self, height=11, width=40)
        self.btnBack = Button(self, text="Return to Main Screen", height="2", width="50", bg="#0066ff", fg="#ffffff",
                              command=ReturnToMain)
        self.btnExit = Button(self, text="Exit", height="2", width="50", bg="#0066ff", fg="#ffffff",
                              command=Close)
        self.createWidgets()

    def createWidgets(self):
        self.Header.grid(row=1, pady=5, sticky=W + E)
        self.information.grid(row=2, pady=5, sticky=W + E)
        self.btnBack.grid(row=3, pady=5, sticky=W + E)
        self.btnExit.grid(row=4, pady=5, sticky=W + E)
        aboutInfo = """

The bot will tell you what type of sickness you have based on your symptoms and the 
doctor data that emerge in relation to your disease analgesics. It will also give you diet suggestions, which indicates what kind of food you should eat. 
The chatbot will ask a series of questions to clarify the user's symptoms, and the symptom confirmation will be completed. 
The sickness will be divided into two categories: minor and major. If it is a significant disease, the chatbot will provide the user with the contact information for a doctor as well as analgesics for further treatment."""

        Creators = """Creators:
Kousani Sarkar (1805127)
Priyansh Choudhary (1805589) 
Shweta Nathany (1806164) 
Debalina Mazumder (1828064)"""
        self.information.insert(tkinter.END, Creators)
        self.information.insert(tkinter.END, aboutInfo)
        self.information.configure(state='disabled')


getSeverityDict()
# print(severityDictionary)
getDescription()
# print(description_list)
getprecautionDict()
# print(precautionDictionary)
getHospitals()
# print(hospitals)
getDoctors()
# getInfo()      info retrieval
window = tkinter.Tk()
# scrollbar = Scrollbar(window)
# scrollbar.pack(side=RIGHT, fill=Y)
# window.grid_columnconfigure(0, weight=1)
MainScreen = MainForm(window)
MainScreen.pack()
window.resizable(False, False)
window.mainloop()
