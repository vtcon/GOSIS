/********************************************************************************
** Form generated from reading UI file 'AddSurfaceDialog.ui'
**
** Created by: Qt User Interface Compiler version 5.12.0
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_ADDSURFACEDIALOG_H
#define UI_ADDSURFACEDIALOG_H

#include <QtCore/QVariant>
#include <QtWidgets/QApplication>
#include <QtWidgets/QDialog>
#include <QtWidgets/QGridLayout>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QLabel>
#include <QtWidgets/QLineEdit>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QSpacerItem>
#include <QtWidgets/QVBoxLayout>

QT_BEGIN_NAMESPACE

class Ui_AddSurfaceDialog
{
public:
    QVBoxLayout *verticalLayout;
    QGridLayout *gridLayout;
    QLabel *label;
    QLineEdit *lineEdit;
    QLabel *label_2;
    QLineEdit *lineEdit_2;
    QLabel *label_3;
    QLineEdit *lineEdit_3;
    QLabel *label_4;
    QLineEdit *lineEdit_4;
    QLabel *label_5;
    QLineEdit *lineEdit_5;
    QLabel *label_6;
    QLineEdit *lineEdit_6;
    QLabel *label_7;
    QLineEdit *lineEdit_7;
    QHBoxLayout *horizontalLayout;
    QSpacerItem *horizontalSpacer;
    QPushButton *pushButton;
    QPushButton *pushButton_2;

    void setupUi(QDialog *AddSurfaceDialog)
    {
        if (AddSurfaceDialog->objectName().isEmpty())
            AddSurfaceDialog->setObjectName(QString::fromUtf8("AddSurfaceDialog"));
        AddSurfaceDialog->resize(400, 300);
        verticalLayout = new QVBoxLayout(AddSurfaceDialog);
        verticalLayout->setSpacing(6);
        verticalLayout->setContentsMargins(11, 11, 11, 11);
        verticalLayout->setObjectName(QString::fromUtf8("verticalLayout"));
        gridLayout = new QGridLayout();
        gridLayout->setSpacing(6);
        gridLayout->setObjectName(QString::fromUtf8("gridLayout"));
        label = new QLabel(AddSurfaceDialog);
        label->setObjectName(QString::fromUtf8("label"));

        gridLayout->addWidget(label, 0, 0, 1, 1);

        lineEdit = new QLineEdit(AddSurfaceDialog);
        lineEdit->setObjectName(QString::fromUtf8("lineEdit"));

        gridLayout->addWidget(lineEdit, 0, 1, 1, 1);

        label_2 = new QLabel(AddSurfaceDialog);
        label_2->setObjectName(QString::fromUtf8("label_2"));

        gridLayout->addWidget(label_2, 1, 0, 1, 1);

        lineEdit_2 = new QLineEdit(AddSurfaceDialog);
        lineEdit_2->setObjectName(QString::fromUtf8("lineEdit_2"));

        gridLayout->addWidget(lineEdit_2, 1, 1, 1, 1);

        label_3 = new QLabel(AddSurfaceDialog);
        label_3->setObjectName(QString::fromUtf8("label_3"));

        gridLayout->addWidget(label_3, 2, 0, 1, 1);

        lineEdit_3 = new QLineEdit(AddSurfaceDialog);
        lineEdit_3->setObjectName(QString::fromUtf8("lineEdit_3"));

        gridLayout->addWidget(lineEdit_3, 2, 1, 1, 1);

        label_4 = new QLabel(AddSurfaceDialog);
        label_4->setObjectName(QString::fromUtf8("label_4"));

        gridLayout->addWidget(label_4, 3, 0, 1, 1);

        lineEdit_4 = new QLineEdit(AddSurfaceDialog);
        lineEdit_4->setObjectName(QString::fromUtf8("lineEdit_4"));

        gridLayout->addWidget(lineEdit_4, 3, 1, 1, 1);

        label_5 = new QLabel(AddSurfaceDialog);
        label_5->setObjectName(QString::fromUtf8("label_5"));

        gridLayout->addWidget(label_5, 4, 0, 1, 1);

        lineEdit_5 = new QLineEdit(AddSurfaceDialog);
        lineEdit_5->setObjectName(QString::fromUtf8("lineEdit_5"));

        gridLayout->addWidget(lineEdit_5, 4, 1, 1, 1);

        label_6 = new QLabel(AddSurfaceDialog);
        label_6->setObjectName(QString::fromUtf8("label_6"));

        gridLayout->addWidget(label_6, 5, 0, 1, 1);

        lineEdit_6 = new QLineEdit(AddSurfaceDialog);
        lineEdit_6->setObjectName(QString::fromUtf8("lineEdit_6"));

        gridLayout->addWidget(lineEdit_6, 5, 1, 1, 1);

        label_7 = new QLabel(AddSurfaceDialog);
        label_7->setObjectName(QString::fromUtf8("label_7"));

        gridLayout->addWidget(label_7, 6, 0, 1, 1);

        lineEdit_7 = new QLineEdit(AddSurfaceDialog);
        lineEdit_7->setObjectName(QString::fromUtf8("lineEdit_7"));

        gridLayout->addWidget(lineEdit_7, 6, 1, 1, 1);


        verticalLayout->addLayout(gridLayout);

        horizontalLayout = new QHBoxLayout();
        horizontalLayout->setSpacing(6);
        horizontalLayout->setObjectName(QString::fromUtf8("horizontalLayout"));
        horizontalSpacer = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout->addItem(horizontalSpacer);

        pushButton = new QPushButton(AddSurfaceDialog);
        pushButton->setObjectName(QString::fromUtf8("pushButton"));

        horizontalLayout->addWidget(pushButton);

        pushButton_2 = new QPushButton(AddSurfaceDialog);
        pushButton_2->setObjectName(QString::fromUtf8("pushButton_2"));

        horizontalLayout->addWidget(pushButton_2);


        verticalLayout->addLayout(horizontalLayout);


        retranslateUi(AddSurfaceDialog);
        QObject::connect(pushButton, SIGNAL(clicked()), AddSurfaceDialog, SLOT(accept()));
        QObject::connect(pushButton_2, SIGNAL(clicked()), AddSurfaceDialog, SLOT(reject()));

        QMetaObject::connectSlotsByName(AddSurfaceDialog);
    } // setupUi

    void retranslateUi(QDialog *AddSurfaceDialog)
    {
        AddSurfaceDialog->setWindowTitle(QApplication::translate("AddSurfaceDialog", "AddSurfaceDialog", nullptr));
        label->setText(QApplication::translate("AddSurfaceDialog", "Surface Vertex X coodinate", nullptr));
        label_2->setText(QApplication::translate("AddSurfaceDialog", "Surface Vertex Y coodinate", nullptr));
        label_3->setText(QApplication::translate("AddSurfaceDialog", "Surface Vertex Z coodinate", nullptr));
        label_4->setText(QApplication::translate("AddSurfaceDialog", "Diameter", nullptr));
        label_5->setText(QApplication::translate("AddSurfaceDialog", "Radius of Curvature", nullptr));
        label_6->setText(QApplication::translate("AddSurfaceDialog", "Refractive Index", nullptr));
        label_7->setText(QApplication::translate("AddSurfaceDialog", "Apherical Coefficient K", nullptr));
        pushButton->setText(QApplication::translate("AddSurfaceDialog", "Add Surface", nullptr));
        pushButton_2->setText(QApplication::translate("AddSurfaceDialog", "Cancel", nullptr));
    } // retranslateUi

};

namespace Ui {
    class AddSurfaceDialog: public Ui_AddSurfaceDialog {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_ADDSURFACEDIALOG_H
