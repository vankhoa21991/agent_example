# Nom et prénom: LE Thi Kim Ngan
# Date: 23/03/2025
# SAE 2.01 - Python Visuel Tkinter - Bataille Navale 2 
# --------------------------------------------------------------

"""
Ce jeu se joue contre l'ordinateur, sur une grille où sont placés 5 navires mis en place par le joueur ainsi que par l'ordinateur, chacun sur sa grille de jeux.
L'ordinateur et le joueur jouent tour à tour.
Le but est de réussir à couler tous les bateaux de l'ordinateur avant qu'il ne coule tous les bateaux du joueur.
Une grille de jeu est constituée de 100 cases numérotées de 1 à 10 horizontalement et de A à J verticalement.
Les bateaux placés sont :
1 x 5 case
1 x 4 cases
2 x 3 cases
1 x 2 cases
"""

"""
On va faire une interface d'affichage graphique des grilles
Dans cette 1ère version : simplement des cases colorées en gris pour les bateaux déposés
* Case colorée en rouge si touché, en bleu clair si à l'eau, blanc sinon
* Placement des bateaux automatique pour le joueur et pour l'ordinateur (apparence par des cases grisées, un gris de couleur différent pour chaque bateau différent). Utiliser par exemple les couleurs suivantes (pour les bateaux) :
couleurs_bateaux=["#222222","#444444","#666666","#888888","#AAAAAA"]
* Saisie des coordonnées de tir pour le joueur dans un Entry
--> afficher un message dans une fenêtre modale si coordonnées incorrectes, attendu 2 caractères: une lettre et un chiffre
(accepter les coordonnées en majuscule ou minuscule, "B4" ou "b4" est correct par exemple)
* Effectue le tour du joueur quand click sur un bouton
* Afficher une information en BLEU et en GRAS du résultat du tir du joueur "A l'eau", "Touché", "Touché coulé !" ou "Déjà fait. Tir gaspillé."
* Effectue le tour de l'ordinateur automatiquement juste après le tour du joueur
* Afficher une information en ROUGE et en GRAS du tir effectué par l'ordinateur et résultat du tir de l'ordinateur "A l'eau", "Touché", "Touché coulé !" ou "Déjà fait. Tir gaspillé."
* Case à cocher "Tricher" qui permet affichage ou pas des cases grisées des bateaux de l'Ordinateur pour tricher, faire en sorte que par défaut le mode triche est activé (case cochée au début)
* Affichage systématique des cases grisées du joueur (de la position de ses bateaux)
* 3 radioboutons permettant de choisir le niveau: choix entre "Très facile" (niveau=0), "Facile" (niveau=1) ou "Normal" (niveau=2). Rendre le boutonradio niveau 2 désactivé (en grisé, pas cochable) car le niveau 2 n'est pas encore implémenté actuellement dans le projet.
Rappel : state="disabled" lors de la construction d'un radiobouton pour le désactiver
* Une boucle effectue successivement le tir humain, puis le tir de l'ordinateur, tant que la paetie n'est pas gagné cela se poursuit indéfiniment.
Après le tir de l'humain, détecter si l'humain a gagné la partie (cela est déjà programmé dans la partie interface textuelle donnée, on compte le nombre de cases de bateau restantes) et afficher une fenêtre avec un showmessage indiquant que l'humain a gagné la partie si c'est le cas. Alors ne pas lancer le tour de l'ordinateur et ne pas répéter la boucle. Lancer le tour de l'ordinateur si l'humain n'a pas gagné. Alors détecter si l'ordinateur a gagné. Si c'est le cas, afficher une fenêtre avec un showmessage indiquant que l'ordinateur a gagné la partie. Et dans ce cas ne pas répéter la boucle.
* Une fois la partie gagnée par le joueur ou l’ordinateur, interdire tout nouveau tir du joueur humain (en cas de tentative de tir, afficher un message indiquant que la partie est terminée et qu’on ne peut pas tirer).

* Mettre un bouton nouvelle partie qui permette de lancer de nouveau une partie en réinitialisant tout (lancer une procédure qui fait cela)
* On souhaite que les tirs successifs du joueur et de l’ordinateur, ainsi que les réponses restent affichés dans le Shell Python en format texte pour le suivi de la partie.

Note : pour changer la police de texte affiché dans un canvas ou un label,
créer d'abord une police personnalisée , puis l'utiliser :

from tkinter import font
police_engras = font.Font(family="Helvetica", size=12, weight="bold")

canvas.create_texte(x,y,text="...",font=police_engras)
Label(fenetre, text="...", font=police_engras)

INDICATION : dans la liste des grilles, remplacer les contenus qui sont des chaines de caractères indiquant le contenu de la case de la grille par un tuple où on met l'information du contenu en 1ère position et ajouter en 2ème position un objet rectangle à créer lors de l'initialisation des grilles.

Note : dans un canvas les objets créés renvoient un numéro d'ordre et les objets créés après s'affichent devant les objets créés avant s’il y a chevauchement.
Donc il faut créer les cases rectangles vides d'initialisation de grille au début
Pour la coloration des cases, penser aussi à colorer les bateaux (en gris) et enfin créer celles de coloration selon tir touché ou à l'eau (qui seront en remplacement des bateaux car créées à la fin)
"""

colonnes = [1,2,3,4,5,6,7,8,9,10]
lignes = ['A','B','C','D','E','F','G','H','I','J']
sens_possibles = ['Horizontal','Vertical']

# Mettre ici les nombres de cases occupées par chaque bateau
# (ordre décroissant préférentiel pour placer les plus gros d'abord)
longueur_bateaux = [5,4,3,3,2]

# Niveau de difficulté du jeu : 0 (très facile) ou 1 (facile)
# Seulement niveaux 0, et 1 implémentés
niveau = 1

# Mode de triche activé (False) ou pas (True)
masquage_grilleordi = False

###########################################################

colonnes_str = [str(valeur) for valeur in colonnes]

from tkinter import *

cote_grille = 400
cote = cote_grille/len(colonnes)

fenetre = Tk()
fenetre.title('Jeu de Bataille Navale')
fenetre.geometry("1060x500")

lj = Label(fenetre, text="Grille du Joueur")
lj.grid(row = 0, column = 0)

lo = Label(fenetre, text="Grille de l'Ordinateur")
lo.grid(row = 0, column = 1)

haut_police = 20
larg_police =  20
from tkinter import font
bold_font = font.Font(family="Arial", size=12, weight="bold")

canvas_joueur = Canvas(fenetre, width = cote_grille+2+larg_police, height = cote_grille+2+haut_police, bg = 'white')
canvas_joueur.grid(row = 1, column = 0, padx = 10, pady = 10)

canvas_ordinateur = Canvas(fenetre, width = cote_grille+2+larg_police, height = cote_grille+2+haut_police, bg = 'white')
canvas_ordinateur.grid(row = 1, column = 1, padx = 10, pady = 10)

## 1
# Procédure de Création des grilles vides avec variable d'entrée/sortie

def initialisation_grille_vide(grille):
    if id(grille)==id(grilleOrdinateur):
        canvas = canvas_ordinateur
    else:
        canvas = canvas_joueur
    for lig in range(len(lignes)):
        lignegrille = []
        for col in range(len(colonnes)):
            rect = canvas.create_rectangle(col*cote+3+larg_police, lig*cote+3+haut_police, (col+1)*cote+3+larg_police, (lig+1)*cote+3+haut_police, width=2, outline='black')
            lignegrille.append(['',rect])
        grille.append(lignegrille)

    for col in range(len(colonnes)):
        canvas.create_text(col*cote+3+cote/2+larg_police, 5, text=str(col+1), anchor=N, fill="blue", font=bold_font)

    for lig in range(len(lignes)):
        canvas.create_text(5, lig*cote+3+haut_police+cote/2, text=str(lignes[lig]), anchor=W, fill="green", font=bold_font)


## 3

# Fonction de test de possibilité de dépôt d'un bateau sur la grille
# Renvoie un booléen, True si on peut déposer le bateau, False sinon
def peut_deposer_bateau(grille,numbateau,ligne,colonne,sens):
    # Longueur du bateau
    longueur = longueur_bateaux[numbateau-1]

    # Test de dépassement
    if sens==0: # si horizontal
        if colonne+longueur-1<=len(colonnes): # a-t-on assez de place à droite
            depassement = False
        else:
            depassement = True
    else: # si vertical
        if ligne+longueur-1<=len(lignes): # a-t-on assez de place en bas
            depassement = False
        else:
            depassement = True

    # Test de cases libres
    if depassement==False:
        occupe = False
        if sens==0: # si horizontal
            colonne_test = colonne
            # colonne variable, ligne fixée, on regarde vers la droite chaque case
            for colonne_test in range(colonne,colonne+longueur):
                if grille[ligne-1][colonne_test-1][0]!="":
                    occupe = True
        else: # si vertical
            ligne_test = ligne
            # ligne variable, colonne fixée, on regarde vers le bas chaque case
            for ligne_test in range(ligne,ligne+longueur):
                if grille[ligne_test-1][colonne-1][0]!="":
                    occupe = True

    # Renvoie True si il n'y a pas de dépassement et qu'il n'y a pas d'occupation de cases : alors on peut placer le bateau
    return not(depassement) and not(occupe)

## 4

# Dépose le bateau sur la grille, en supposant que le dépôt est possible
# Donc avoir testé préalablement qu'on peut déposer le bateau sinon plantage
def depose_bateau(grille,numbateau,ligne,colonne,sens):
    longueur = longueur_bateaux[numbateau-1]

    if sens==0:
        colonne_test = colonne
        for colonne_test in range(colonne,colonne+longueur):
            grille[ligne-1][colonne_test-1][0]=numbateau
    else:
        ligne_test = ligne
        for ligne_test in range(ligne,ligne+longueur):
            grille[ligne_test-1][colonne-1][0]=numbateau

## 5
# Dépôt des bateaux de l'Ordinateur

from random import randint
# Dépose les bateaux dont les nombres de cases sont définis
# par 'longueur_bateaux' de manière aléatoire dans la grille
# en faisant en sorte que le dépôt soit possible (pas de collision
# de bateaux ni de dépassement hors grille)
def depot_aleatoire_bateaux(grille):

    # On parcourt chaque bateau de l'ordinateur pour le placer
    for numbateau_aplacer in range(1,len(longueur_bateaux)+1):

        test = False

        # Boucle infinie tant qu'on n'a pas trouvé une solution de placement
        # La boucle ne sera pas infinie car sur une grille 10x10
        # il est impossible d'avoir une situation de placement des bateaux
        # qui puisse bloquer le placement des autres
        # Il suffira de laisser le hasard explorer assez longtemps
        # et il trouvera toujours une solution
        while test==False:
            # Tirage d'une case de début de dépôt
            ligne_alea = randint(1,len(lignes))
            colonne_alea = randint(1,len(colonnes))

            # Tirage d'un sens de dépôt
            # sens_alea = 0 : horizontal / sens_alea = 1 : vertical
            sens_alea = randint(0,1)

            # Peut-on déposer le bateau ?
            test = peut_deposer_bateau(grille,numbateau_aplacer,ligne_alea,colonne_alea,sens_alea)

        # Quand on sort de la boucle c'est qu'on a trouvé une solution de placement, alors on dépose le bateau là où on a dit que c'est possible
        depose_bateau(grille,numbateau_aplacer,ligne_alea,colonne_alea,sens_alea)
        # Information pour le programmeur (pour tests)
        # A enlever en version finale du jeu
        # print("Longueur bateau : ",longueur_bateaux[numbateau_aplacer-1]," - Début en : ",lignes[ligne_alea-1],colonne_alea," - Sens : ",sens_possibles[sens_alea],sep="")




## 6
# Affichage de la grille au format texte (interface visuelle à faire ensuite)

def traduit(lettre):
    return lignes.index(lettre)

couleurs_bateaux=["#222222","#444444","#666666","#888888","#AAAAAA"]
# Dessine le contenu d'une case de canvas avec une apparence
# dépendant de valeur, aux coordonnées l et c
def trace(val,rect,can):
    if val=="X":
        couleur="red"
    elif val=="o":
        couleur="lightblue"
    elif val==".":
        couleur="white"
    elif int(val) in list(range(1,len(longueur_bateaux)+1)):
        couleur=couleurs_bateaux[int(val)-1]
    can.itemconfigure(rect,fill=couleur)

# Affiche la grille avec les bateaux ou en les masquant
def affiche_grille(grille,masquebateaux):
    if id(grille)==id(grilleOrdinateur):
        canvas = canvas_ordinateur
    else:
        canvas = canvas_joueur
    for lig in lignes:
        for col in colonnes:
            ll = traduit(lig)
            cc = col-1
            valeur = grille[ll][cc][0]
            rect = grille[ll][cc][1]
            if valeur=="":
                valeur="."
            if masquebateaux and valeur in list(range(1,len(longueur_bateaux)+1)):
                valeur="."
            trace(valeur,rect,canvas)


## 9

# (interface visuelle à faire)
# Saisir un tir du joueur
def saisie_tir_joueur():
    # Choix d'une case de début de dépôt par le joueur humain
    ligne_joueur = ""
    while not(ligne_joueur in lignes): # Boucle jusqu'à saisie correcte
        ligne_joueur = input("Ligne ('A' à 'J'): ")
        if not(ligne_joueur in lignes):
            print("Saisie incorrecte. Recommencez !")

    ligne_joueur = traduit(ligne_joueur)+1 # ligne comme entier

    colonne_joueur = ""
    colonnes_str = [str(valeur) for valeur in colonnes]
    while not(colonne_joueur in colonnes_str):
        colonne_joueur = input("Colonne ('1' à '10'): ")
        if not(colonne_joueur in colonnes_str):
            print("Saisie incorrecte. Recommencez !")

    colonne_joueur = int(colonne_joueur) # colonne comme entier

    return (ligne_joueur,colonne_joueur)

## 10
# Tours de jeu (interface visuelle à faire)



# Effectue le tir dans la grille et décompte le tir des cases des bateaux
# Renvoie un booléen True si fin de partie (plus de bateau restants) et un coderesultat
def effectue_le_tir_sur_grille(ligne,colonne,grille,bateaux,masquage):
    contenucase = grille[ligne-1][colonne-1][0]
    if contenucase=="":
        print("--> A l'eau")
        coderesultat = 0
        grille[ligne-1][colonne-1][0]="o"
    elif contenucase=="X":
        print("--> Tir gaspillé. Déjà touché auparavant.")
        coderesultat = -1
    elif contenucase=="o":
        print("--> Tir gaspillé. Déjà à l'eau auparavant.")
        coderesultat = -2
    else:
        print("--> Touché",end="")
        coderesultat = 1
        grille[ligne-1][colonne-1][0]="X"
        indexbateau = contenucase-1
        bateaux[indexbateau] = bateaux[indexbateau]-1
        if bateaux[indexbateau]==0:
            print("- Coulé")
            coderesultat = 2
        else:
            print()

    # On compte le nombre de cases de bateaux non coulés restant
    nbrestant = sum(bateaux)
    return (nbrestant==0,coderesultat) # renvoie True si plus aucun bateau restant


# Décision de l'ordinateur sur où tirer le tir suivant
def choix_tir_suivant(niv,listedejatires):
    if niv==0:
        # Choix aléatoire d'une case de tir de l'ordinateur
        ligne = randint(1,len(lignes))
        colonne = randint(1,len(colonnes))
    elif niv==1:
        # Choix aléatoire d'une case de tir de l'ordinateur
        accepte = False
        while not(accepte):
            ligne = randint(1,len(lignes))
            colonne = randint(1,len(colonnes))
            if (ligne,colonne) in listedejatires:
                accepte = False
            else:
                listedejatires.append((ligne,colonne))
                accepte = True

    return (ligne,colonne)


# Renvoie une information texte correspondante au code
def renvoie_infocode(code):
    if code==1:
        texteinfo="Touché"
    elif code==0:
        texteinfo="A l'eau"
    elif code==2:
        texteinfo="Touché coulé !"
    else:
        texteinfo="Déjà fait. Tir gaspillé."

    return texteinfo

from tkinter.messagebox import showinfo

# Lit la saisie de tir effectuée par le joueur
def lit_tir():
    global findepartie
    if not(findepartie):
        # Tour du Joueur

        # Activer ce code pour jouer vraiment
        coord_tir = tirjoueurvar.get()

        if len(coord_tir)==2 or len(coord_tir)==3:
            coord_tir = coord_tir.upper()
            if (coord_tir[0] in lignes) and (coord_tir[1:] in colonnes_str):
                ligne = lignes.index(coord_tir[0])+1
                colonne = int(coord_tir[1:])

                evenement_tir_joueur(ligne,colonne)
            else:
                showinfo(title='Tir non effectué', message='Coordonnées de tir incorrectes. Recommencer.')
        else:
            showinfo(title='Tir non effectué', message='Les coordonnées de tir doivent comporter exactement 2 ou 3 caractères. Recommencer.')
    else:
        showinfo(title='Tir non effectué', message='Le jeu est terminé. Vous ne pouvez plus tirer.')

def evenement_tir_joueur(ligne_joueur_tir,colonne_joueur_tir):
    global findepartie
    # Tour du Joueur

    # Désactiver cette ligne pour éviter le jeu automatique
    # Jeu en niveau 0 pour le joueur
    # ligne_joueur_tir,colonne_joueur_tir=choix_tir_suivant(0,listedejatires_joueur)
    print("Joueur - Tir en : ",lignes[ligne_joueur_tir-1],colonne_joueur_tir,sep="")

    listedejatires_joueur.append((ligne_joueur_tir,colonne_joueur_tir))
    (findepartie,code) = effectue_le_tir_sur_grille(ligne_joueur_tir,colonne_joueur_tir,grilleOrdinateur,bateaux_ordinateur,True)

    lreschoix_joueur.config(text=lignes[ligne_joueur_tir-1]+str(colonne_joueur_tir))
    lres_joueur.config(text=renvoie_infocode(code))
    affiche_grille(grilleOrdinateur,masquage_grilleordi)
    if findepartie:
        print("Le joueur a gagné la partie! Fin du jeu.")
        showinfo(title='Fin de partie', message="Le joueur a gagné la partie! Fin du jeu.")
    else:
        # Tour de l'ordinateur

        # Jeu en niveau donné par la variable "niveau" pour l'ordinateur
        ligne_ordi_tir,colonne_alea_tir=choix_tir_suivant(niveau,listedejatires_ordinateur)

        print("Ordinateur - Tir en : ",lignes[ligne_ordi_tir-1],colonne_alea_tir,sep="")
        listedejatires_ordinateur.append((ligne_ordi_tir,colonne_alea_tir))
        (findepartie,code) = effectue_le_tir_sur_grille(ligne_ordi_tir,colonne_alea_tir,grilleJoueur,bateaux_joueur,False)

        lres_ordi.config(text=renvoie_infocode(code))
        lo2.config(text=lignes[ligne_ordi_tir-1]+str(colonne_alea_tir))
        affiche_grille(grilleJoueur,False)
        if findepartie:
            print("L'ordinateur a gagné la partie! Fin du jeu.")
            showinfo(title='Fin de partie', message="L'ordinateur a gagné la partie! Fin du jeu.")

#############
def nouvelle_partie():
    global grilleJoueur,grilleOrdinateur
    grilleJoueur = []
    grilleOrdinateur = []

    initialisation_grille_vide(grilleJoueur)
    initialisation_grille_vide(grilleOrdinateur)

    print("Constitution aléatoire des bâteaux sur la grille Ordinateur")
    depot_aleatoire_bateaux(grilleOrdinateur)

    # Affichage de la grille de l'ordinateur
    affiche_grille(grilleOrdinateur,masquage_grilleordi)
    print()

    # Saisie des bateaux par l'utilisateur (interface visuelle à faire) : manuelle ou aléatoire automatique

    # Tirage aléatoire des bateaux pour le joueur
    print("Constitution aléatoire des bâteaux sur la grille du Joueur")
    depot_aleatoire_bateaux(grilleJoueur)
    affiche_grille(grilleJoueur,False)
    print()

    # Attention, shallow_copy, copie profonde
    # car il ne faut pas que la liste des bateaux de l'ordinateur
    # soir aussi celle du joueur, il faut 2 listes indépendantes
    global bateaux_ordinateur,bateaux_joueur
    bateaux_ordinateur = longueur_bateaux[:]
    bateaux_joueur = longueur_bateaux[:]

    global listedejatires_joueur,listedejatires_ordinateur
    listedejatires_joueur = []
    listedejatires_ordinateur = []

    global findepartie
    findepartie = False

    lres_ordi.config(text="")
    lreschoix_joueur.config(text="")
    lres_joueur.config(text="")
    lo2.config(text="")

#

frameoutil = Frame(fenetre)
frameoutil.grid(row = 1, column = 2, sticky="NSEW")

#

frameinfo_tirordi = Frame(frameoutil)
frameinfo_tirordi.pack(side=TOP)

def actualiser():
    etattriche = modetriche.get()
    global masquage_grilleordi
    masquage_grilleordi = etattriche==0
    affiche_grille(grilleOrdinateur,masquage_grilleordi)

modetriche = IntVar()
if masquage_grilleordi==False:
    modetriche.set(1)
else:
    modetriche.set(0)
cb = Checkbutton(frameinfo_tirordi,variable=modetriche,text="Tricher",command=actualiser)
cb.grid(row = 0, column = 0)

lo1 = Label(frameinfo_tirordi, text="L'ordinateur a tiré en :", fg="red")
lo1.grid(row = 1, column = 0)
lo2 = Label(frameinfo_tirordi, text="", fg="red", font=bold_font)
lo2.grid(row = 2, column = 0)

lres_ordi = Label(frameinfo_tirordi, text="", fg="red", font=bold_font)
lres_ordi.grid(row = 3, column = 0)

#

framenew = Frame(frameoutil)
framenew.pack(side=TOP)

bnew = Button(framenew, text="Nouvelle partie", command = nouvelle_partie)
bnew.grid(row = 0, column = 0)

lencours = Label(framenew, text="Régler le niveau IA\nde la partie en cours :")
lencours.grid(row = 1, column = 0, pady=20)

def changerleniveau():
    global niveau
    niveau = leniveau.get()

leniveau = IntVar()
leniveau.set(niveau)

rb1 = Radiobutton(framenew,variable=leniveau,value=0,text="Très facile",command=changerleniveau)
rb1.grid(row = 2, column = 0)

rb2 = Radiobutton(framenew,variable=leniveau,value=1,text="Facile",command=changerleniveau)
rb2.grid(row = 3, column = 0)

rb3 = Radiobutton(framenew,variable=leniveau,value=2,text="Normal",command=changerleniveau, state="disabled")
rb3.grid(row = 4, column = 0)

#

frametir = Frame(frameoutil)
frametir.pack(side=BOTTOM)

l = Label(frametir, text="Coordonnées de tir :")
l.grid(row = 0, column = 0)

tirjoueurvar = StringVar()
ej = Entry(frametir, textvariable=tirjoueurvar, width=3)
ej.grid(row = 1, column = 0)

b = Button(frametir, text="Effectuer le tir", command = lit_tir)
b.grid(row = 2, column = 0)

l2 = Label(frametir, text="Résultat du tir du joueur:", fg="blue")
l2.grid(row = 3, column = 0, pady=10)

lreschoix_joueur = Label(frametir, text="", fg="blue", font=bold_font)
lreschoix_joueur.grid(row = 4, column = 0)

lres_joueur = Label(frametir, text="", fg="blue", font=bold_font)
lres_joueur.grid(row = 5, column = 0)

nouvelle_partie()
fenetre.mainloop()