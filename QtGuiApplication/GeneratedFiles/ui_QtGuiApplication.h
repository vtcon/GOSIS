/********************************************************************************
** Form generated from reading UI file 'QtGuiApplication.ui'
**
** Created by: Qt User Interface Compiler version 5.12.0
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_QTGUIAPPLICATION_H
#define UI_QTGUIAPPLICATION_H

#include <QtCore/QVariant>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QCheckBox>
#include <QtWidgets/QGridLayout>
#include <QtWidgets/QGroupBox>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QLabel>
#include <QtWidgets/QLineEdit>
#include <QtWidgets/QListWidget>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QMenu>
#include <QtWidgets/QMenuBar>
#include <QtWidgets/QPlainTextEdit>
#include <QtWidgets/QProgressBar>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QRadioButton>
#include <QtWidgets/QSpacerItem>
#include <QtWidgets/QStackedWidget>
#include <QtWidgets/QStatusBar>
#include <QtWidgets/QTabWidget>
#include <QtWidgets/QTableWidget>
#include <QtWidgets/QTextEdit>
#include <QtWidgets/QToolBar>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_QtGuiApplicationClass
{
public:
    QAction *actionNew;
    QAction *actionOpen;
    QAction *actionSave;
    QAction *actionSave_as;
    QAction *actionClose;
    QAction *actionPreferences;
    QAction *actionSystem_Info;
    QAction *actionDocumentation;
    QAction *actionHelp;
    QAction *actionTest;
    QAction *actionConsoleOut;
    QWidget *centralWidget;
    QVBoxLayout *verticalLayout;
    QTabWidget *tabWidget;
    QWidget *tabInput;
    QGridLayout *gridLayout_2;
    QPushButton *pushClearInput;
    QSpacerItem *verticalSpacer_2;
    QStackedWidget *stackedInput;
    QWidget *pageManual;
    QGridLayout *gridLayout;
    QTableWidget *tableInput;
    QPushButton *pushAddPoint;
    QPushButton *pushRemovePoint;
    QSpacerItem *verticalSpacer;
    QWidget *pagePicture;
    QGridLayout *gridLayout_6;
    QGridLayout *gridLayout_14;
    QLabel *label_23;
    QLineEdit *lineImagePath;
    QPushButton *pushSelectImage;
    QPushButton *pushClearImage;
    QGroupBox *groupBox_2;
    QGridLayout *gridLayout_3;
    QLabel *label_5;
    QLineEdit *lineImagePosX;
    QLabel *label_6;
    QLineEdit *lineImagePosY;
    QLabel *label_7;
    QLineEdit *lineImagePosZ;
    QGridLayout *gridLayout_15;
    QLabel *label_24;
    QLineEdit *lineImageBrightness;
    QGroupBox *groupBox_8;
    QGridLayout *gridLayout_5;
    QLabel *label_11;
    QLineEdit *lineImageHorzSize;
    QLabel *label_18;
    QLineEdit *lineImageVertSize;
    QGroupBox *groupBox_3;
    QGridLayout *gridLayout_4;
    QLabel *label_8;
    QLineEdit *lineImageRotX;
    QLabel *label_9;
    QLineEdit *lineImageRotY;
    QLabel *label_10;
    QLineEdit *lineImageRotZ;
    QLabel *label_22;
    QWidget *pageList;
    QGridLayout *gridLayout_7;
    QHBoxLayout *horizontalLayout_3;
    QPushButton *pushButton_3;
    QLineEdit *lineEdit_8;
    QPushButton *pushButton_4;
    QSpacerItem *horizontalSpacer_4;
    QVBoxLayout *verticalLayout_3;
    QLabel *label_12;
    QPlainTextEdit *plainTextEdit_2;
    QSpacerItem *verticalSpacer_6;
    QSpacerItem *verticalSpacer_5;
    QGroupBox *groupBox;
    QVBoxLayout *verticalLayout_2;
    QRadioButton *radioManual;
    QRadioButton *radioPicture;
    QRadioButton *radioList;
    QWidget *tabConfig;
    QGridLayout *gridLayout_16;
    QGroupBox *groupBox_4;
    QVBoxLayout *verticalLayout_4;
    QListWidget *listConfig;
    QGroupBox *groupBox_6;
    QGridLayout *gridLayout_10;
    QGridLayout *gridLayout_8;
    QSpacerItem *horizontalSpacer_2;
    QPushButton *pushAddSurface;
    QPushButton *pushModifySurface;
    QPushButton *pushRemoveSurface;
    QTableWidget *tableConfig;
    QSpacerItem *horizontalSpacer_6;
    QPushButton *pushAcceptConfig;
    QPushButton *pushClearConfig;
    QPushButton *pushVisualizeConfig;
    QPushButton *pushCloneConfig;
    QPushButton *pushSaveConfig;
    QPushButton *pushLoadConfig;
    QGroupBox *groupBox_7;
    QGridLayout *gridLayout_9;
    QLabel *label_16;
    QLineEdit *lineImageDiam;
    QLabel *label_14;
    QLineEdit *lineAngularResol;
    QLineEdit *lineAngularExtend;
    QLabel *label_13;
    QLabel *label_15;
    QLineEdit *lineImageRadius;
    QWidget *tabProcessAndOutput;
    QGridLayout *gridLayout_13;
    QGroupBox *groupBox_5;
    QGridLayout *gridLayout_11;
    QPushButton *pushCheckData;
    QPushButton *pushTraceAndRender;
    QSpacerItem *horizontalSpacer_3;
    QLabel *labelTime;
    QPushButton *pushPause;
    QPushButton *pushTrace;
    QProgressBar *progressTrace;
    QPushButton *pushRender;
    QProgressBar *progressRender;
    QTextEdit *textEditProcess;
    QGroupBox *groupBox_9;
    QGridLayout *gridLayout_12;
    QLabel *label_17;
    QListWidget *listOutputWavelength;
    QHBoxLayout *horizontalLayout_2;
    QPushButton *pushShowWavelength;
    QPushButton *pushDisplayRGB;
    QHBoxLayout *horizontalLayout_5;
    QPushButton *pushVisualizeRGB;
    QCheckBox *checkShowOpticalSurfaces;
    QHBoxLayout *horizontalLayout_4;
    QPushButton *pushSaveRaw;
    QPushButton *pushSaveRGB;
    QHBoxLayout *horizontalLayout;
    QLabel *label;
    QLabel *label_2;
    QSpacerItem *horizontalSpacer;
    QLabel *label_3;
    QLabel *labelResUsage;
    QMenuBar *menuBar;
    QMenu *menuSession;
    QMenu *menuHelp;
    QToolBar *mainToolBar;
    QStatusBar *statusBar;

    void setupUi(QMainWindow *QtGuiApplicationClass)
    {
        if (QtGuiApplicationClass->objectName().isEmpty())
            QtGuiApplicationClass->setObjectName(QString::fromUtf8("QtGuiApplicationClass"));
        QtGuiApplicationClass->resize(961, 577);
        actionNew = new QAction(QtGuiApplicationClass);
        actionNew->setObjectName(QString::fromUtf8("actionNew"));
        actionOpen = new QAction(QtGuiApplicationClass);
        actionOpen->setObjectName(QString::fromUtf8("actionOpen"));
        actionSave = new QAction(QtGuiApplicationClass);
        actionSave->setObjectName(QString::fromUtf8("actionSave"));
        actionSave_as = new QAction(QtGuiApplicationClass);
        actionSave_as->setObjectName(QString::fromUtf8("actionSave_as"));
        actionClose = new QAction(QtGuiApplicationClass);
        actionClose->setObjectName(QString::fromUtf8("actionClose"));
        actionPreferences = new QAction(QtGuiApplicationClass);
        actionPreferences->setObjectName(QString::fromUtf8("actionPreferences"));
        actionSystem_Info = new QAction(QtGuiApplicationClass);
        actionSystem_Info->setObjectName(QString::fromUtf8("actionSystem_Info"));
        actionDocumentation = new QAction(QtGuiApplicationClass);
        actionDocumentation->setObjectName(QString::fromUtf8("actionDocumentation"));
        actionHelp = new QAction(QtGuiApplicationClass);
        actionHelp->setObjectName(QString::fromUtf8("actionHelp"));
        actionTest = new QAction(QtGuiApplicationClass);
        actionTest->setObjectName(QString::fromUtf8("actionTest"));
        actionConsoleOut = new QAction(QtGuiApplicationClass);
        actionConsoleOut->setObjectName(QString::fromUtf8("actionConsoleOut"));
        centralWidget = new QWidget(QtGuiApplicationClass);
        centralWidget->setObjectName(QString::fromUtf8("centralWidget"));
        verticalLayout = new QVBoxLayout(centralWidget);
        verticalLayout->setSpacing(6);
        verticalLayout->setContentsMargins(11, 11, 11, 11);
        verticalLayout->setObjectName(QString::fromUtf8("verticalLayout"));
        tabWidget = new QTabWidget(centralWidget);
        tabWidget->setObjectName(QString::fromUtf8("tabWidget"));
        tabInput = new QWidget();
        tabInput->setObjectName(QString::fromUtf8("tabInput"));
        gridLayout_2 = new QGridLayout(tabInput);
        gridLayout_2->setSpacing(6);
        gridLayout_2->setContentsMargins(11, 11, 11, 11);
        gridLayout_2->setObjectName(QString::fromUtf8("gridLayout_2"));
        pushClearInput = new QPushButton(tabInput);
        pushClearInput->setObjectName(QString::fromUtf8("pushClearInput"));

        gridLayout_2->addWidget(pushClearInput, 2, 0, 1, 1);

        verticalSpacer_2 = new QSpacerItem(20, 188, QSizePolicy::Minimum, QSizePolicy::Expanding);

        gridLayout_2->addItem(verticalSpacer_2, 1, 0, 1, 1);

        stackedInput = new QStackedWidget(tabInput);
        stackedInput->setObjectName(QString::fromUtf8("stackedInput"));
        pageManual = new QWidget();
        pageManual->setObjectName(QString::fromUtf8("pageManual"));
        gridLayout = new QGridLayout(pageManual);
        gridLayout->setSpacing(6);
        gridLayout->setContentsMargins(11, 11, 11, 11);
        gridLayout->setObjectName(QString::fromUtf8("gridLayout"));
        tableInput = new QTableWidget(pageManual);
        if (tableInput->columnCount() < 4)
            tableInput->setColumnCount(4);
        QTableWidgetItem *__qtablewidgetitem = new QTableWidgetItem();
        tableInput->setHorizontalHeaderItem(0, __qtablewidgetitem);
        QTableWidgetItem *__qtablewidgetitem1 = new QTableWidgetItem();
        tableInput->setHorizontalHeaderItem(1, __qtablewidgetitem1);
        QTableWidgetItem *__qtablewidgetitem2 = new QTableWidgetItem();
        tableInput->setHorizontalHeaderItem(2, __qtablewidgetitem2);
        QTableWidgetItem *__qtablewidgetitem3 = new QTableWidgetItem();
        tableInput->setHorizontalHeaderItem(3, __qtablewidgetitem3);
        tableInput->setObjectName(QString::fromUtf8("tableInput"));
        tableInput->setEditTriggers(QAbstractItemView::NoEditTriggers);
        tableInput->setSelectionMode(QAbstractItemView::SingleSelection);
        tableInput->setSelectionBehavior(QAbstractItemView::SelectRows);
        tableInput->setRowCount(0);
        tableInput->horizontalHeader()->setStretchLastSection(true);

        gridLayout->addWidget(tableInput, 0, 0, 3, 1);

        pushAddPoint = new QPushButton(pageManual);
        pushAddPoint->setObjectName(QString::fromUtf8("pushAddPoint"));

        gridLayout->addWidget(pushAddPoint, 0, 1, 1, 1);

        pushRemovePoint = new QPushButton(pageManual);
        pushRemovePoint->setObjectName(QString::fromUtf8("pushRemovePoint"));

        gridLayout->addWidget(pushRemovePoint, 1, 1, 1, 1);

        verticalSpacer = new QSpacerItem(20, 287, QSizePolicy::Minimum, QSizePolicy::Expanding);

        gridLayout->addItem(verticalSpacer, 2, 1, 1, 1);

        stackedInput->addWidget(pageManual);
        pagePicture = new QWidget();
        pagePicture->setObjectName(QString::fromUtf8("pagePicture"));
        gridLayout_6 = new QGridLayout(pagePicture);
        gridLayout_6->setSpacing(6);
        gridLayout_6->setContentsMargins(11, 11, 11, 11);
        gridLayout_6->setObjectName(QString::fromUtf8("gridLayout_6"));
        gridLayout_14 = new QGridLayout();
        gridLayout_14->setSpacing(6);
        gridLayout_14->setObjectName(QString::fromUtf8("gridLayout_14"));
        label_23 = new QLabel(pagePicture);
        label_23->setObjectName(QString::fromUtf8("label_23"));

        gridLayout_14->addWidget(label_23, 0, 0, 1, 1);

        lineImagePath = new QLineEdit(pagePicture);
        lineImagePath->setObjectName(QString::fromUtf8("lineImagePath"));
        lineImagePath->setReadOnly(true);

        gridLayout_14->addWidget(lineImagePath, 0, 1, 1, 1);

        pushSelectImage = new QPushButton(pagePicture);
        pushSelectImage->setObjectName(QString::fromUtf8("pushSelectImage"));

        gridLayout_14->addWidget(pushSelectImage, 0, 2, 1, 1);

        pushClearImage = new QPushButton(pagePicture);
        pushClearImage->setObjectName(QString::fromUtf8("pushClearImage"));

        gridLayout_14->addWidget(pushClearImage, 0, 3, 1, 1);


        gridLayout_6->addLayout(gridLayout_14, 0, 0, 1, 2);

        groupBox_2 = new QGroupBox(pagePicture);
        groupBox_2->setObjectName(QString::fromUtf8("groupBox_2"));
        gridLayout_3 = new QGridLayout(groupBox_2);
        gridLayout_3->setSpacing(6);
        gridLayout_3->setContentsMargins(11, 11, 11, 11);
        gridLayout_3->setObjectName(QString::fromUtf8("gridLayout_3"));
        label_5 = new QLabel(groupBox_2);
        label_5->setObjectName(QString::fromUtf8("label_5"));

        gridLayout_3->addWidget(label_5, 0, 0, 1, 1);

        lineImagePosX = new QLineEdit(groupBox_2);
        lineImagePosX->setObjectName(QString::fromUtf8("lineImagePosX"));

        gridLayout_3->addWidget(lineImagePosX, 0, 1, 1, 1);

        label_6 = new QLabel(groupBox_2);
        label_6->setObjectName(QString::fromUtf8("label_6"));

        gridLayout_3->addWidget(label_6, 1, 0, 1, 1);

        lineImagePosY = new QLineEdit(groupBox_2);
        lineImagePosY->setObjectName(QString::fromUtf8("lineImagePosY"));

        gridLayout_3->addWidget(lineImagePosY, 1, 1, 1, 1);

        label_7 = new QLabel(groupBox_2);
        label_7->setObjectName(QString::fromUtf8("label_7"));

        gridLayout_3->addWidget(label_7, 2, 0, 1, 1);

        lineImagePosZ = new QLineEdit(groupBox_2);
        lineImagePosZ->setObjectName(QString::fromUtf8("lineImagePosZ"));

        gridLayout_3->addWidget(lineImagePosZ, 2, 1, 1, 1);


        gridLayout_6->addWidget(groupBox_2, 1, 0, 2, 1);

        gridLayout_15 = new QGridLayout();
        gridLayout_15->setSpacing(6);
        gridLayout_15->setObjectName(QString::fromUtf8("gridLayout_15"));
        label_24 = new QLabel(pagePicture);
        label_24->setObjectName(QString::fromUtf8("label_24"));

        gridLayout_15->addWidget(label_24, 0, 0, 1, 1);

        lineImageBrightness = new QLineEdit(pagePicture);
        lineImageBrightness->setObjectName(QString::fromUtf8("lineImageBrightness"));

        gridLayout_15->addWidget(lineImageBrightness, 0, 1, 1, 1);


        gridLayout_6->addLayout(gridLayout_15, 1, 1, 1, 1);

        groupBox_8 = new QGroupBox(pagePicture);
        groupBox_8->setObjectName(QString::fromUtf8("groupBox_8"));
        gridLayout_5 = new QGridLayout(groupBox_8);
        gridLayout_5->setSpacing(6);
        gridLayout_5->setContentsMargins(11, 11, 11, 11);
        gridLayout_5->setObjectName(QString::fromUtf8("gridLayout_5"));
        label_11 = new QLabel(groupBox_8);
        label_11->setObjectName(QString::fromUtf8("label_11"));

        gridLayout_5->addWidget(label_11, 0, 0, 1, 1);

        lineImageHorzSize = new QLineEdit(groupBox_8);
        lineImageHorzSize->setObjectName(QString::fromUtf8("lineImageHorzSize"));

        gridLayout_5->addWidget(lineImageHorzSize, 0, 1, 1, 1);

        label_18 = new QLabel(groupBox_8);
        label_18->setObjectName(QString::fromUtf8("label_18"));

        gridLayout_5->addWidget(label_18, 1, 0, 1, 1);

        lineImageVertSize = new QLineEdit(groupBox_8);
        lineImageVertSize->setObjectName(QString::fromUtf8("lineImageVertSize"));

        gridLayout_5->addWidget(lineImageVertSize, 1, 1, 1, 1);


        gridLayout_6->addWidget(groupBox_8, 2, 1, 2, 1);

        groupBox_3 = new QGroupBox(pagePicture);
        groupBox_3->setObjectName(QString::fromUtf8("groupBox_3"));
        gridLayout_4 = new QGridLayout(groupBox_3);
        gridLayout_4->setSpacing(6);
        gridLayout_4->setContentsMargins(11, 11, 11, 11);
        gridLayout_4->setObjectName(QString::fromUtf8("gridLayout_4"));
        label_8 = new QLabel(groupBox_3);
        label_8->setObjectName(QString::fromUtf8("label_8"));

        gridLayout_4->addWidget(label_8, 0, 0, 1, 1);

        lineImageRotX = new QLineEdit(groupBox_3);
        lineImageRotX->setObjectName(QString::fromUtf8("lineImageRotX"));

        gridLayout_4->addWidget(lineImageRotX, 0, 1, 1, 1);

        label_9 = new QLabel(groupBox_3);
        label_9->setObjectName(QString::fromUtf8("label_9"));

        gridLayout_4->addWidget(label_9, 1, 0, 1, 1);

        lineImageRotY = new QLineEdit(groupBox_3);
        lineImageRotY->setObjectName(QString::fromUtf8("lineImageRotY"));

        gridLayout_4->addWidget(lineImageRotY, 1, 1, 1, 1);

        label_10 = new QLabel(groupBox_3);
        label_10->setObjectName(QString::fromUtf8("label_10"));

        gridLayout_4->addWidget(label_10, 2, 0, 1, 1);

        lineImageRotZ = new QLineEdit(groupBox_3);
        lineImageRotZ->setObjectName(QString::fromUtf8("lineImageRotZ"));

        gridLayout_4->addWidget(lineImageRotZ, 2, 1, 1, 1);

        label_22 = new QLabel(groupBox_3);
        label_22->setObjectName(QString::fromUtf8("label_22"));

        gridLayout_4->addWidget(label_22, 3, 0, 1, 2);


        gridLayout_6->addWidget(groupBox_3, 3, 0, 1, 1);

        stackedInput->addWidget(pagePicture);
        pageList = new QWidget();
        pageList->setObjectName(QString::fromUtf8("pageList"));
        gridLayout_7 = new QGridLayout(pageList);
        gridLayout_7->setSpacing(6);
        gridLayout_7->setContentsMargins(11, 11, 11, 11);
        gridLayout_7->setObjectName(QString::fromUtf8("gridLayout_7"));
        horizontalLayout_3 = new QHBoxLayout();
        horizontalLayout_3->setSpacing(6);
        horizontalLayout_3->setObjectName(QString::fromUtf8("horizontalLayout_3"));
        pushButton_3 = new QPushButton(pageList);
        pushButton_3->setObjectName(QString::fromUtf8("pushButton_3"));

        horizontalLayout_3->addWidget(pushButton_3);

        lineEdit_8 = new QLineEdit(pageList);
        lineEdit_8->setObjectName(QString::fromUtf8("lineEdit_8"));

        horizontalLayout_3->addWidget(lineEdit_8);


        gridLayout_7->addLayout(horizontalLayout_3, 0, 0, 1, 1);

        pushButton_4 = new QPushButton(pageList);
        pushButton_4->setObjectName(QString::fromUtf8("pushButton_4"));

        gridLayout_7->addWidget(pushButton_4, 0, 1, 1, 1);

        horizontalSpacer_4 = new QSpacerItem(147, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        gridLayout_7->addItem(horizontalSpacer_4, 0, 2, 2, 1);

        verticalLayout_3 = new QVBoxLayout();
        verticalLayout_3->setSpacing(6);
        verticalLayout_3->setObjectName(QString::fromUtf8("verticalLayout_3"));
        label_12 = new QLabel(pageList);
        label_12->setObjectName(QString::fromUtf8("label_12"));

        verticalLayout_3->addWidget(label_12);

        plainTextEdit_2 = new QPlainTextEdit(pageList);
        plainTextEdit_2->setObjectName(QString::fromUtf8("plainTextEdit_2"));

        verticalLayout_3->addWidget(plainTextEdit_2);


        gridLayout_7->addLayout(verticalLayout_3, 1, 0, 2, 1);

        verticalSpacer_6 = new QSpacerItem(20, 283, QSizePolicy::Minimum, QSizePolicy::Expanding);

        gridLayout_7->addItem(verticalSpacer_6, 2, 1, 2, 1);

        verticalSpacer_5 = new QSpacerItem(20, 171, QSizePolicy::Minimum, QSizePolicy::Expanding);

        gridLayout_7->addItem(verticalSpacer_5, 3, 0, 1, 1);

        stackedInput->addWidget(pageList);

        gridLayout_2->addWidget(stackedInput, 0, 1, 3, 1);

        groupBox = new QGroupBox(tabInput);
        groupBox->setObjectName(QString::fromUtf8("groupBox"));
        verticalLayout_2 = new QVBoxLayout(groupBox);
        verticalLayout_2->setSpacing(6);
        verticalLayout_2->setContentsMargins(11, 11, 11, 11);
        verticalLayout_2->setObjectName(QString::fromUtf8("verticalLayout_2"));
        radioManual = new QRadioButton(groupBox);
        radioManual->setObjectName(QString::fromUtf8("radioManual"));
        radioManual->setChecked(true);

        verticalLayout_2->addWidget(radioManual);

        radioPicture = new QRadioButton(groupBox);
        radioPicture->setObjectName(QString::fromUtf8("radioPicture"));

        verticalLayout_2->addWidget(radioPicture);

        radioList = new QRadioButton(groupBox);
        radioList->setObjectName(QString::fromUtf8("radioList"));

        verticalLayout_2->addWidget(radioList);


        gridLayout_2->addWidget(groupBox, 0, 0, 1, 1);

        tabWidget->addTab(tabInput, QString());
        tabConfig = new QWidget();
        tabConfig->setObjectName(QString::fromUtf8("tabConfig"));
        gridLayout_16 = new QGridLayout(tabConfig);
        gridLayout_16->setSpacing(6);
        gridLayout_16->setContentsMargins(11, 11, 11, 11);
        gridLayout_16->setObjectName(QString::fromUtf8("gridLayout_16"));
        groupBox_4 = new QGroupBox(tabConfig);
        groupBox_4->setObjectName(QString::fromUtf8("groupBox_4"));
        verticalLayout_4 = new QVBoxLayout(groupBox_4);
        verticalLayout_4->setSpacing(6);
        verticalLayout_4->setContentsMargins(11, 11, 11, 11);
        verticalLayout_4->setObjectName(QString::fromUtf8("verticalLayout_4"));
        listConfig = new QListWidget(groupBox_4);
        listConfig->setObjectName(QString::fromUtf8("listConfig"));
        QSizePolicy sizePolicy(QSizePolicy::Minimum, QSizePolicy::Expanding);
        sizePolicy.setHorizontalStretch(0);
        sizePolicy.setVerticalStretch(0);
        sizePolicy.setHeightForWidth(listConfig->sizePolicy().hasHeightForWidth());
        listConfig->setSizePolicy(sizePolicy);
        listConfig->setEditTriggers(QAbstractItemView::NoEditTriggers);
        listConfig->setAlternatingRowColors(false);
        listConfig->setSelectionRectVisible(true);

        verticalLayout_4->addWidget(listConfig);


        gridLayout_16->addWidget(groupBox_4, 0, 0, 1, 1);

        groupBox_6 = new QGroupBox(tabConfig);
        groupBox_6->setObjectName(QString::fromUtf8("groupBox_6"));
        QSizePolicy sizePolicy1(QSizePolicy::Expanding, QSizePolicy::Preferred);
        sizePolicy1.setHorizontalStretch(0);
        sizePolicy1.setVerticalStretch(0);
        sizePolicy1.setHeightForWidth(groupBox_6->sizePolicy().hasHeightForWidth());
        groupBox_6->setSizePolicy(sizePolicy1);
        gridLayout_10 = new QGridLayout(groupBox_6);
        gridLayout_10->setSpacing(6);
        gridLayout_10->setContentsMargins(11, 11, 11, 11);
        gridLayout_10->setObjectName(QString::fromUtf8("gridLayout_10"));
        gridLayout_8 = new QGridLayout();
        gridLayout_8->setSpacing(6);
        gridLayout_8->setObjectName(QString::fromUtf8("gridLayout_8"));
        horizontalSpacer_2 = new QSpacerItem(238, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        gridLayout_8->addItem(horizontalSpacer_2, 0, 0, 1, 1);

        pushAddSurface = new QPushButton(groupBox_6);
        pushAddSurface->setObjectName(QString::fromUtf8("pushAddSurface"));

        gridLayout_8->addWidget(pushAddSurface, 0, 1, 1, 1);

        pushModifySurface = new QPushButton(groupBox_6);
        pushModifySurface->setObjectName(QString::fromUtf8("pushModifySurface"));

        gridLayout_8->addWidget(pushModifySurface, 0, 2, 1, 1);

        pushRemoveSurface = new QPushButton(groupBox_6);
        pushRemoveSurface->setObjectName(QString::fromUtf8("pushRemoveSurface"));

        gridLayout_8->addWidget(pushRemoveSurface, 0, 3, 1, 1);


        gridLayout_10->addLayout(gridLayout_8, 0, 0, 1, 7);

        tableConfig = new QTableWidget(groupBox_6);
        if (tableConfig->columnCount() < 6)
            tableConfig->setColumnCount(6);
        QTableWidgetItem *__qtablewidgetitem4 = new QTableWidgetItem();
        tableConfig->setHorizontalHeaderItem(0, __qtablewidgetitem4);
        QTableWidgetItem *__qtablewidgetitem5 = new QTableWidgetItem();
        tableConfig->setHorizontalHeaderItem(1, __qtablewidgetitem5);
        QTableWidgetItem *__qtablewidgetitem6 = new QTableWidgetItem();
        tableConfig->setHorizontalHeaderItem(2, __qtablewidgetitem6);
        QTableWidgetItem *__qtablewidgetitem7 = new QTableWidgetItem();
        tableConfig->setHorizontalHeaderItem(3, __qtablewidgetitem7);
        QTableWidgetItem *__qtablewidgetitem8 = new QTableWidgetItem();
        tableConfig->setHorizontalHeaderItem(4, __qtablewidgetitem8);
        QTableWidgetItem *__qtablewidgetitem9 = new QTableWidgetItem();
        tableConfig->setHorizontalHeaderItem(5, __qtablewidgetitem9);
        tableConfig->setObjectName(QString::fromUtf8("tableConfig"));
        tableConfig->setSelectionBehavior(QAbstractItemView::SelectRows);

        gridLayout_10->addWidget(tableConfig, 1, 0, 1, 7);

        horizontalSpacer_6 = new QSpacerItem(87, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        gridLayout_10->addItem(horizontalSpacer_6, 2, 0, 1, 1);

        pushAcceptConfig = new QPushButton(groupBox_6);
        pushAcceptConfig->setObjectName(QString::fromUtf8("pushAcceptConfig"));
        QFont font;
        font.setBold(true);
        font.setWeight(75);
        pushAcceptConfig->setFont(font);

        gridLayout_10->addWidget(pushAcceptConfig, 2, 1, 1, 1);

        pushClearConfig = new QPushButton(groupBox_6);
        pushClearConfig->setObjectName(QString::fromUtf8("pushClearConfig"));

        gridLayout_10->addWidget(pushClearConfig, 2, 2, 1, 1);

        pushVisualizeConfig = new QPushButton(groupBox_6);
        pushVisualizeConfig->setObjectName(QString::fromUtf8("pushVisualizeConfig"));

        gridLayout_10->addWidget(pushVisualizeConfig, 2, 3, 1, 1);

        pushCloneConfig = new QPushButton(groupBox_6);
        pushCloneConfig->setObjectName(QString::fromUtf8("pushCloneConfig"));

        gridLayout_10->addWidget(pushCloneConfig, 2, 4, 1, 1);

        pushSaveConfig = new QPushButton(groupBox_6);
        pushSaveConfig->setObjectName(QString::fromUtf8("pushSaveConfig"));

        gridLayout_10->addWidget(pushSaveConfig, 2, 5, 1, 1);

        pushLoadConfig = new QPushButton(groupBox_6);
        pushLoadConfig->setObjectName(QString::fromUtf8("pushLoadConfig"));

        gridLayout_10->addWidget(pushLoadConfig, 2, 6, 1, 1);


        gridLayout_16->addWidget(groupBox_6, 0, 1, 2, 1);

        groupBox_7 = new QGroupBox(tabConfig);
        groupBox_7->setObjectName(QString::fromUtf8("groupBox_7"));
        gridLayout_9 = new QGridLayout(groupBox_7);
        gridLayout_9->setSpacing(6);
        gridLayout_9->setContentsMargins(11, 11, 11, 11);
        gridLayout_9->setObjectName(QString::fromUtf8("gridLayout_9"));
        label_16 = new QLabel(groupBox_7);
        label_16->setObjectName(QString::fromUtf8("label_16"));

        gridLayout_9->addWidget(label_16, 3, 0, 1, 1);

        lineImageDiam = new QLineEdit(groupBox_7);
        lineImageDiam->setObjectName(QString::fromUtf8("lineImageDiam"));
        QSizePolicy sizePolicy2(QSizePolicy::Preferred, QSizePolicy::Fixed);
        sizePolicy2.setHorizontalStretch(0);
        sizePolicy2.setVerticalStretch(0);
        sizePolicy2.setHeightForWidth(lineImageDiam->sizePolicy().hasHeightForWidth());
        lineImageDiam->setSizePolicy(sizePolicy2);

        gridLayout_9->addWidget(lineImageDiam, 0, 1, 1, 1);

        label_14 = new QLabel(groupBox_7);
        label_14->setObjectName(QString::fromUtf8("label_14"));

        gridLayout_9->addWidget(label_14, 1, 0, 1, 1);

        lineAngularResol = new QLineEdit(groupBox_7);
        lineAngularResol->setObjectName(QString::fromUtf8("lineAngularResol"));
        sizePolicy2.setHeightForWidth(lineAngularResol->sizePolicy().hasHeightForWidth());
        lineAngularResol->setSizePolicy(sizePolicy2);

        gridLayout_9->addWidget(lineAngularResol, 2, 1, 1, 1);

        lineAngularExtend = new QLineEdit(groupBox_7);
        lineAngularExtend->setObjectName(QString::fromUtf8("lineAngularExtend"));
        sizePolicy2.setHeightForWidth(lineAngularExtend->sizePolicy().hasHeightForWidth());
        lineAngularExtend->setSizePolicy(sizePolicy2);

        gridLayout_9->addWidget(lineAngularExtend, 3, 1, 1, 1);

        label_13 = new QLabel(groupBox_7);
        label_13->setObjectName(QString::fromUtf8("label_13"));

        gridLayout_9->addWidget(label_13, 0, 0, 1, 1);

        label_15 = new QLabel(groupBox_7);
        label_15->setObjectName(QString::fromUtf8("label_15"));

        gridLayout_9->addWidget(label_15, 2, 0, 1, 1);

        lineImageRadius = new QLineEdit(groupBox_7);
        lineImageRadius->setObjectName(QString::fromUtf8("lineImageRadius"));
        sizePolicy2.setHeightForWidth(lineImageRadius->sizePolicy().hasHeightForWidth());
        lineImageRadius->setSizePolicy(sizePolicy2);

        gridLayout_9->addWidget(lineImageRadius, 1, 1, 1, 1);


        gridLayout_16->addWidget(groupBox_7, 1, 0, 1, 1);

        tabWidget->addTab(tabConfig, QString());
        tabProcessAndOutput = new QWidget();
        tabProcessAndOutput->setObjectName(QString::fromUtf8("tabProcessAndOutput"));
        gridLayout_13 = new QGridLayout(tabProcessAndOutput);
        gridLayout_13->setSpacing(6);
        gridLayout_13->setContentsMargins(11, 11, 11, 11);
        gridLayout_13->setObjectName(QString::fromUtf8("gridLayout_13"));
        groupBox_5 = new QGroupBox(tabProcessAndOutput);
        groupBox_5->setObjectName(QString::fromUtf8("groupBox_5"));
        gridLayout_11 = new QGridLayout(groupBox_5);
        gridLayout_11->setSpacing(6);
        gridLayout_11->setContentsMargins(11, 11, 11, 11);
        gridLayout_11->setObjectName(QString::fromUtf8("gridLayout_11"));
        pushCheckData = new QPushButton(groupBox_5);
        pushCheckData->setObjectName(QString::fromUtf8("pushCheckData"));

        gridLayout_11->addWidget(pushCheckData, 0, 0, 1, 1);

        pushTraceAndRender = new QPushButton(groupBox_5);
        pushTraceAndRender->setObjectName(QString::fromUtf8("pushTraceAndRender"));

        gridLayout_11->addWidget(pushTraceAndRender, 0, 1, 1, 1);

        horizontalSpacer_3 = new QSpacerItem(378, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        gridLayout_11->addItem(horizontalSpacer_3, 0, 2, 1, 1);

        labelTime = new QLabel(groupBox_5);
        labelTime->setObjectName(QString::fromUtf8("labelTime"));

        gridLayout_11->addWidget(labelTime, 0, 3, 1, 1);

        pushPause = new QPushButton(groupBox_5);
        pushPause->setObjectName(QString::fromUtf8("pushPause"));

        gridLayout_11->addWidget(pushPause, 0, 4, 1, 1);

        pushTrace = new QPushButton(groupBox_5);
        pushTrace->setObjectName(QString::fromUtf8("pushTrace"));

        gridLayout_11->addWidget(pushTrace, 1, 0, 1, 1);

        progressTrace = new QProgressBar(groupBox_5);
        progressTrace->setObjectName(QString::fromUtf8("progressTrace"));
        progressTrace->setValue(0);

        gridLayout_11->addWidget(progressTrace, 1, 1, 1, 4);

        pushRender = new QPushButton(groupBox_5);
        pushRender->setObjectName(QString::fromUtf8("pushRender"));

        gridLayout_11->addWidget(pushRender, 2, 0, 1, 1);

        progressRender = new QProgressBar(groupBox_5);
        progressRender->setObjectName(QString::fromUtf8("progressRender"));
        progressRender->setValue(0);

        gridLayout_11->addWidget(progressRender, 2, 1, 1, 4);

        textEditProcess = new QTextEdit(groupBox_5);
        textEditProcess->setObjectName(QString::fromUtf8("textEditProcess"));

        gridLayout_11->addWidget(textEditProcess, 3, 0, 1, 5);


        gridLayout_13->addWidget(groupBox_5, 0, 0, 1, 1);

        groupBox_9 = new QGroupBox(tabProcessAndOutput);
        groupBox_9->setObjectName(QString::fromUtf8("groupBox_9"));
        gridLayout_12 = new QGridLayout(groupBox_9);
        gridLayout_12->setSpacing(6);
        gridLayout_12->setContentsMargins(11, 11, 11, 11);
        gridLayout_12->setObjectName(QString::fromUtf8("gridLayout_12"));
        label_17 = new QLabel(groupBox_9);
        label_17->setObjectName(QString::fromUtf8("label_17"));

        gridLayout_12->addWidget(label_17, 0, 0, 1, 1);

        listOutputWavelength = new QListWidget(groupBox_9);
        listOutputWavelength->setObjectName(QString::fromUtf8("listOutputWavelength"));
        QSizePolicy sizePolicy3(QSizePolicy::Fixed, QSizePolicy::Expanding);
        sizePolicy3.setHorizontalStretch(0);
        sizePolicy3.setVerticalStretch(0);
        sizePolicy3.setHeightForWidth(listOutputWavelength->sizePolicy().hasHeightForWidth());
        listOutputWavelength->setSizePolicy(sizePolicy3);
        listOutputWavelength->setSelectionMode(QAbstractItemView::SingleSelection);

        gridLayout_12->addWidget(listOutputWavelength, 1, 0, 1, 1);

        horizontalLayout_2 = new QHBoxLayout();
        horizontalLayout_2->setSpacing(6);
        horizontalLayout_2->setObjectName(QString::fromUtf8("horizontalLayout_2"));
        pushShowWavelength = new QPushButton(groupBox_9);
        pushShowWavelength->setObjectName(QString::fromUtf8("pushShowWavelength"));

        horizontalLayout_2->addWidget(pushShowWavelength);

        pushDisplayRGB = new QPushButton(groupBox_9);
        pushDisplayRGB->setObjectName(QString::fromUtf8("pushDisplayRGB"));

        horizontalLayout_2->addWidget(pushDisplayRGB);


        gridLayout_12->addLayout(horizontalLayout_2, 2, 0, 1, 1);

        horizontalLayout_5 = new QHBoxLayout();
        horizontalLayout_5->setSpacing(6);
        horizontalLayout_5->setObjectName(QString::fromUtf8("horizontalLayout_5"));
        pushVisualizeRGB = new QPushButton(groupBox_9);
        pushVisualizeRGB->setObjectName(QString::fromUtf8("pushVisualizeRGB"));

        horizontalLayout_5->addWidget(pushVisualizeRGB);

        checkShowOpticalSurfaces = new QCheckBox(groupBox_9);
        checkShowOpticalSurfaces->setObjectName(QString::fromUtf8("checkShowOpticalSurfaces"));

        horizontalLayout_5->addWidget(checkShowOpticalSurfaces);


        gridLayout_12->addLayout(horizontalLayout_5, 3, 0, 1, 1);

        horizontalLayout_4 = new QHBoxLayout();
        horizontalLayout_4->setSpacing(6);
        horizontalLayout_4->setObjectName(QString::fromUtf8("horizontalLayout_4"));
        pushSaveRaw = new QPushButton(groupBox_9);
        pushSaveRaw->setObjectName(QString::fromUtf8("pushSaveRaw"));

        horizontalLayout_4->addWidget(pushSaveRaw);

        pushSaveRGB = new QPushButton(groupBox_9);
        pushSaveRGB->setObjectName(QString::fromUtf8("pushSaveRGB"));

        horizontalLayout_4->addWidget(pushSaveRGB);


        gridLayout_12->addLayout(horizontalLayout_4, 4, 0, 1, 1);


        gridLayout_13->addWidget(groupBox_9, 0, 1, 1, 1);

        tabWidget->addTab(tabProcessAndOutput, QString());

        verticalLayout->addWidget(tabWidget);

        horizontalLayout = new QHBoxLayout();
        horizontalLayout->setSpacing(6);
        horizontalLayout->setObjectName(QString::fromUtf8("horizontalLayout"));
        label = new QLabel(centralWidget);
        label->setObjectName(QString::fromUtf8("label"));

        horizontalLayout->addWidget(label);

        label_2 = new QLabel(centralWidget);
        label_2->setObjectName(QString::fromUtf8("label_2"));

        horizontalLayout->addWidget(label_2);

        horizontalSpacer = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout->addItem(horizontalSpacer);

        label_3 = new QLabel(centralWidget);
        label_3->setObjectName(QString::fromUtf8("label_3"));

        horizontalLayout->addWidget(label_3);

        labelResUsage = new QLabel(centralWidget);
        labelResUsage->setObjectName(QString::fromUtf8("labelResUsage"));
        sizePolicy1.setHeightForWidth(labelResUsage->sizePolicy().hasHeightForWidth());
        labelResUsage->setSizePolicy(sizePolicy1);

        horizontalLayout->addWidget(labelResUsage);


        verticalLayout->addLayout(horizontalLayout);

        QtGuiApplicationClass->setCentralWidget(centralWidget);
        menuBar = new QMenuBar(QtGuiApplicationClass);
        menuBar->setObjectName(QString::fromUtf8("menuBar"));
        menuBar->setGeometry(QRect(0, 0, 961, 26));
        menuSession = new QMenu(menuBar);
        menuSession->setObjectName(QString::fromUtf8("menuSession"));
        menuHelp = new QMenu(menuBar);
        menuHelp->setObjectName(QString::fromUtf8("menuHelp"));
        QtGuiApplicationClass->setMenuBar(menuBar);
        mainToolBar = new QToolBar(QtGuiApplicationClass);
        mainToolBar->setObjectName(QString::fromUtf8("mainToolBar"));
        QtGuiApplicationClass->addToolBar(Qt::TopToolBarArea, mainToolBar);
        statusBar = new QStatusBar(QtGuiApplicationClass);
        statusBar->setObjectName(QString::fromUtf8("statusBar"));
        QtGuiApplicationClass->setStatusBar(statusBar);

        menuBar->addAction(menuSession->menuAction());
        menuBar->addAction(menuHelp->menuAction());
        menuSession->addAction(actionNew);
        menuSession->addAction(actionOpen);
        menuSession->addSeparator();
        menuSession->addAction(actionSave);
        menuSession->addAction(actionSave_as);
        menuSession->addSeparator();
        menuSession->addAction(actionClose);
        menuSession->addSeparator();
        menuSession->addAction(actionConsoleOut);
        menuHelp->addAction(actionPreferences);
        menuHelp->addAction(actionSystem_Info);
        menuHelp->addSeparator();
        menuHelp->addAction(actionDocumentation);
        menuHelp->addAction(actionHelp);
        menuHelp->addAction(actionTest);

        retranslateUi(QtGuiApplicationClass);

        tabWidget->setCurrentIndex(0);
        stackedInput->setCurrentIndex(0);
        pushAcceptConfig->setDefault(true);


        QMetaObject::connectSlotsByName(QtGuiApplicationClass);
    } // setupUi

    void retranslateUi(QMainWindow *QtGuiApplicationClass)
    {
        QtGuiApplicationClass->setWindowTitle(QApplication::translate("QtGuiApplicationClass", "QtGuiApplication", nullptr));
        actionNew->setText(QApplication::translate("QtGuiApplicationClass", "New", nullptr));
        actionOpen->setText(QApplication::translate("QtGuiApplicationClass", "Open", nullptr));
        actionSave->setText(QApplication::translate("QtGuiApplicationClass", "Save", nullptr));
        actionSave_as->setText(QApplication::translate("QtGuiApplicationClass", "Save as...", nullptr));
        actionClose->setText(QApplication::translate("QtGuiApplicationClass", "Close", nullptr));
        actionPreferences->setText(QApplication::translate("QtGuiApplicationClass", "Preferences", nullptr));
        actionSystem_Info->setText(QApplication::translate("QtGuiApplicationClass", "System Info", nullptr));
        actionDocumentation->setText(QApplication::translate("QtGuiApplicationClass", "Documentation", nullptr));
        actionHelp->setText(QApplication::translate("QtGuiApplicationClass", "About", nullptr));
        actionTest->setText(QApplication::translate("QtGuiApplicationClass", "Test", nullptr));
        actionConsoleOut->setText(QApplication::translate("QtGuiApplicationClass", "Console Out", nullptr));
        pushClearInput->setText(QApplication::translate("QtGuiApplicationClass", "Clear All", nullptr));
        QTableWidgetItem *___qtablewidgetitem = tableInput->horizontalHeaderItem(0);
        ___qtablewidgetitem->setText(QApplication::translate("QtGuiApplicationClass", "Coordinate", nullptr));
        QTableWidgetItem *___qtablewidgetitem1 = tableInput->horizontalHeaderItem(1);
        ___qtablewidgetitem1->setText(QApplication::translate("QtGuiApplicationClass", "Wavelength", nullptr));
        QTableWidgetItem *___qtablewidgetitem2 = tableInput->horizontalHeaderItem(2);
        ___qtablewidgetitem2->setText(QApplication::translate("QtGuiApplicationClass", "Intensity", nullptr));
        QTableWidgetItem *___qtablewidgetitem3 = tableInput->horizontalHeaderItem(3);
        ___qtablewidgetitem3->setText(QApplication::translate("QtGuiApplicationClass", "UniqueID", nullptr));
        pushAddPoint->setText(QApplication::translate("QtGuiApplicationClass", "Add Point", nullptr));
        pushRemovePoint->setText(QApplication::translate("QtGuiApplicationClass", "Remove Point", nullptr));
        label_23->setText(QApplication::translate("QtGuiApplicationClass", "Image Path:", nullptr));
        pushSelectImage->setText(QApplication::translate("QtGuiApplicationClass", "Select Image", nullptr));
        pushClearImage->setText(QApplication::translate("QtGuiApplicationClass", "Clear", nullptr));
        groupBox_2->setTitle(QApplication::translate("QtGuiApplicationClass", "World Position Of Image (Top-Left Corner)", nullptr));
        label_5->setText(QApplication::translate("QtGuiApplicationClass", "X Coordinate", nullptr));
        lineImagePosX->setText(QApplication::translate("QtGuiApplicationClass", "15", nullptr));
        label_6->setText(QApplication::translate("QtGuiApplicationClass", "Y Coordinate", nullptr));
        lineImagePosY->setText(QApplication::translate("QtGuiApplicationClass", "10", nullptr));
        label_7->setText(QApplication::translate("QtGuiApplicationClass", "Z Coordinate", nullptr));
        lineImagePosZ->setText(QApplication::translate("QtGuiApplicationClass", "250", nullptr));
        label_24->setText(QApplication::translate("QtGuiApplicationClass", "Image Brightness", nullptr));
        lineImageBrightness->setText(QApplication::translate("QtGuiApplicationClass", "1.0", nullptr));
        groupBox_8->setTitle(QApplication::translate("QtGuiApplicationClass", "Size Of Image", nullptr));
        label_11->setText(QApplication::translate("QtGuiApplicationClass", "Horizontal Size", nullptr));
        lineImageHorzSize->setText(QApplication::translate("QtGuiApplicationClass", "30", nullptr));
        label_18->setText(QApplication::translate("QtGuiApplicationClass", "Vertical Size", nullptr));
        lineImageVertSize->setText(QApplication::translate("QtGuiApplicationClass", "20", nullptr));
        groupBox_3->setTitle(QApplication::translate("QtGuiApplicationClass", "World Orientation Of Image (Degree)", nullptr));
        label_8->setText(QApplication::translate("QtGuiApplicationClass", "X Rotation", nullptr));
        lineImageRotX->setText(QApplication::translate("QtGuiApplicationClass", "0.0", nullptr));
        label_9->setText(QApplication::translate("QtGuiApplicationClass", "Y Rotation", nullptr));
        lineImageRotY->setText(QApplication::translate("QtGuiApplicationClass", "0.0", nullptr));
        label_10->setText(QApplication::translate("QtGuiApplicationClass", "Z Rotation", nullptr));
        lineImageRotZ->setText(QApplication::translate("QtGuiApplicationClass", "0.0", nullptr));
        label_22->setText(QApplication::translate("QtGuiApplicationClass", "Note: Rotation will be applied in the order Z -> Y -> X", nullptr));
        pushButton_3->setText(QApplication::translate("QtGuiApplicationClass", "Select File:", nullptr));
        pushButton_4->setText(QApplication::translate("QtGuiApplicationClass", "Import From File", nullptr));
        label_12->setText(QApplication::translate("QtGuiApplicationClass", "Info", nullptr));
        groupBox->setTitle(QApplication::translate("QtGuiApplicationClass", "Input Mode", nullptr));
        radioManual->setText(QApplication::translate("QtGuiApplicationClass", "Enter Points Manually", nullptr));
        radioPicture->setText(QApplication::translate("QtGuiApplicationClass", "Load From Picture", nullptr));
        radioList->setText(QApplication::translate("QtGuiApplicationClass", "Load From Point List", nullptr));
        tabWidget->setTabText(tabWidget->indexOf(tabInput), QApplication::translate("QtGuiApplicationClass", "Input", nullptr));
        groupBox_4->setTitle(QApplication::translate("QtGuiApplicationClass", "Wavelength:", nullptr));
        groupBox_6->setTitle(QApplication::translate("QtGuiApplicationClass", "Optical Surfaces At Selected Wavelength:", nullptr));
        pushAddSurface->setText(QApplication::translate("QtGuiApplicationClass", "Add Surface", nullptr));
        pushModifySurface->setText(QApplication::translate("QtGuiApplicationClass", "Modify Surface", nullptr));
        pushRemoveSurface->setText(QApplication::translate("QtGuiApplicationClass", "Remove Surface", nullptr));
        QTableWidgetItem *___qtablewidgetitem4 = tableConfig->horizontalHeaderItem(0);
        ___qtablewidgetitem4->setText(QApplication::translate("QtGuiApplicationClass", "Coordinate", nullptr));
        QTableWidgetItem *___qtablewidgetitem5 = tableConfig->horizontalHeaderItem(1);
        ___qtablewidgetitem5->setText(QApplication::translate("QtGuiApplicationClass", "Diameter", nullptr));
        QTableWidgetItem *___qtablewidgetitem6 = tableConfig->horizontalHeaderItem(2);
        ___qtablewidgetitem6->setText(QApplication::translate("QtGuiApplicationClass", "Curvature Radius", nullptr));
        QTableWidgetItem *___qtablewidgetitem7 = tableConfig->horizontalHeaderItem(3);
        ___qtablewidgetitem7->setText(QApplication::translate("QtGuiApplicationClass", "Refractive Index", nullptr));
        QTableWidgetItem *___qtablewidgetitem8 = tableConfig->horizontalHeaderItem(4);
        ___qtablewidgetitem8->setText(QApplication::translate("QtGuiApplicationClass", "Asphericity", nullptr));
        QTableWidgetItem *___qtablewidgetitem9 = tableConfig->horizontalHeaderItem(5);
        ___qtablewidgetitem9->setText(QApplication::translate("QtGuiApplicationClass", "Apodization", nullptr));
        pushAcceptConfig->setText(QApplication::translate("QtGuiApplicationClass", "Accept", nullptr));
        pushClearConfig->setText(QApplication::translate("QtGuiApplicationClass", "Clear", nullptr));
        pushVisualizeConfig->setText(QApplication::translate("QtGuiApplicationClass", "Visualize", nullptr));
        pushCloneConfig->setText(QApplication::translate("QtGuiApplicationClass", "Clone...", nullptr));
        pushSaveConfig->setText(QApplication::translate("QtGuiApplicationClass", "Save...", nullptr));
        pushLoadConfig->setText(QApplication::translate("QtGuiApplicationClass", "Load...", nullptr));
        groupBox_7->setTitle(QApplication::translate("QtGuiApplicationClass", "Image Surface", nullptr));
        label_16->setText(QApplication::translate("QtGuiApplicationClass", "Max Angular Extend (Deg)", nullptr));
        lineImageDiam->setText(QApplication::translate("QtGuiApplicationClass", "40.0", nullptr));
        label_14->setText(QApplication::translate("QtGuiApplicationClass", "Curvature Radius", nullptr));
        lineAngularResol->setText(QApplication::translate("QtGuiApplicationClass", "0.16", nullptr));
        lineAngularExtend->setText(QApplication::translate("QtGuiApplicationClass", "90.0", nullptr));
        label_13->setText(QApplication::translate("QtGuiApplicationClass", "Diameter", nullptr));
        label_15->setText(QApplication::translate("QtGuiApplicationClass", "Angular Resolution (Deg)", nullptr));
        lineImageRadius->setText(QApplication::translate("QtGuiApplicationClass", "-60.0", nullptr));
        tabWidget->setTabText(tabWidget->indexOf(tabConfig), QApplication::translate("QtGuiApplicationClass", "Configuration", nullptr));
        groupBox_5->setTitle(QApplication::translate("QtGuiApplicationClass", "Processing", nullptr));
        pushCheckData->setText(QApplication::translate("QtGuiApplicationClass", "Data Check-in", nullptr));
        pushTraceAndRender->setText(QApplication::translate("QtGuiApplicationClass", "Trace and Render", nullptr));
        labelTime->setText(QApplication::translate("QtGuiApplicationClass", "TextLabel", nullptr));
        pushPause->setText(QApplication::translate("QtGuiApplicationClass", "Pause", nullptr));
        pushTrace->setText(QApplication::translate("QtGuiApplicationClass", "Trace", nullptr));
        pushRender->setText(QApplication::translate("QtGuiApplicationClass", "Render", nullptr));
        groupBox_9->setTitle(QApplication::translate("QtGuiApplicationClass", "Output", nullptr));
        label_17->setText(QApplication::translate("QtGuiApplicationClass", "Available Wavelengths", nullptr));
        pushShowWavelength->setText(QApplication::translate("QtGuiApplicationClass", "Show Wavelength", nullptr));
        pushDisplayRGB->setText(QApplication::translate("QtGuiApplicationClass", "Display RGB", nullptr));
        pushVisualizeRGB->setText(QApplication::translate("QtGuiApplicationClass", "Visualize RGB", nullptr));
        checkShowOpticalSurfaces->setText(QApplication::translate("QtGuiApplicationClass", "show optical surfaces", nullptr));
        pushSaveRaw->setText(QApplication::translate("QtGuiApplicationClass", "Save Raw", nullptr));
        pushSaveRGB->setText(QApplication::translate("QtGuiApplicationClass", "Save RGB", nullptr));
        tabWidget->setTabText(tabWidget->indexOf(tabProcessAndOutput), QApplication::translate("QtGuiApplicationClass", "Processing And Output", nullptr));
        label->setText(QApplication::translate("QtGuiApplicationClass", "Active Session:", nullptr));
        label_2->setText(QApplication::translate("QtGuiApplicationClass", "<Session name>", nullptr));
        label_3->setText(QApplication::translate("QtGuiApplicationClass", "Resource Usage:", nullptr));
        labelResUsage->setText(QApplication::translate("QtGuiApplicationClass", "<RAM and VRAM>", nullptr));
        menuSession->setTitle(QApplication::translate("QtGuiApplicationClass", "Session", nullptr));
        menuHelp->setTitle(QApplication::translate("QtGuiApplicationClass", "Help", nullptr));
    } // retranslateUi

};

namespace Ui {
    class QtGuiApplicationClass: public Ui_QtGuiApplicationClass {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_QTGUIAPPLICATION_H
