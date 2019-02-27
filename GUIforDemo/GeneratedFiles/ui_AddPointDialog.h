/********************************************************************************
** Form generated from reading UI file 'AddPointDialog.ui'
**
** Created by: Qt User Interface Compiler version 5.12.0
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_ADDPOINTDIALOG_H
#define UI_ADDPOINTDIALOG_H

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

class Ui_AddPointDialog
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
    QHBoxLayout *horizontalLayout;
    QSpacerItem *horizontalSpacer;
    QPushButton *pushButton;
    QPushButton *pushButton_2;

    void setupUi(QDialog *AddPointDialog)
    {
        if (AddPointDialog->objectName().isEmpty())
            AddPointDialog->setObjectName(QString::fromUtf8("AddPointDialog"));
        AddPointDialog->resize(400, 243);
        verticalLayout = new QVBoxLayout(AddPointDialog);
        verticalLayout->setSpacing(6);
        verticalLayout->setContentsMargins(11, 11, 11, 11);
        verticalLayout->setObjectName(QString::fromUtf8("verticalLayout"));
        gridLayout = new QGridLayout();
        gridLayout->setSpacing(6);
        gridLayout->setObjectName(QString::fromUtf8("gridLayout"));
        label = new QLabel(AddPointDialog);
        label->setObjectName(QString::fromUtf8("label"));

        gridLayout->addWidget(label, 0, 0, 1, 1);

        lineEdit = new QLineEdit(AddPointDialog);
        lineEdit->setObjectName(QString::fromUtf8("lineEdit"));

        gridLayout->addWidget(lineEdit, 0, 1, 1, 1);

        label_2 = new QLabel(AddPointDialog);
        label_2->setObjectName(QString::fromUtf8("label_2"));

        gridLayout->addWidget(label_2, 1, 0, 1, 1);

        lineEdit_2 = new QLineEdit(AddPointDialog);
        lineEdit_2->setObjectName(QString::fromUtf8("lineEdit_2"));

        gridLayout->addWidget(lineEdit_2, 1, 1, 1, 1);

        label_3 = new QLabel(AddPointDialog);
        label_3->setObjectName(QString::fromUtf8("label_3"));

        gridLayout->addWidget(label_3, 2, 0, 1, 1);

        lineEdit_3 = new QLineEdit(AddPointDialog);
        lineEdit_3->setObjectName(QString::fromUtf8("lineEdit_3"));

        gridLayout->addWidget(lineEdit_3, 2, 1, 1, 1);

        label_4 = new QLabel(AddPointDialog);
        label_4->setObjectName(QString::fromUtf8("label_4"));

        gridLayout->addWidget(label_4, 3, 0, 1, 1);

        lineEdit_4 = new QLineEdit(AddPointDialog);
        lineEdit_4->setObjectName(QString::fromUtf8("lineEdit_4"));

        gridLayout->addWidget(lineEdit_4, 3, 1, 1, 1);

        label_5 = new QLabel(AddPointDialog);
        label_5->setObjectName(QString::fromUtf8("label_5"));

        gridLayout->addWidget(label_5, 4, 0, 1, 1);

        lineEdit_5 = new QLineEdit(AddPointDialog);
        lineEdit_5->setObjectName(QString::fromUtf8("lineEdit_5"));

        gridLayout->addWidget(lineEdit_5, 4, 1, 1, 1);


        verticalLayout->addLayout(gridLayout);

        horizontalLayout = new QHBoxLayout();
        horizontalLayout->setSpacing(6);
        horizontalLayout->setObjectName(QString::fromUtf8("horizontalLayout"));
        horizontalSpacer = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout->addItem(horizontalSpacer);

        pushButton = new QPushButton(AddPointDialog);
        pushButton->setObjectName(QString::fromUtf8("pushButton"));

        horizontalLayout->addWidget(pushButton);

        pushButton_2 = new QPushButton(AddPointDialog);
        pushButton_2->setObjectName(QString::fromUtf8("pushButton_2"));

        horizontalLayout->addWidget(pushButton_2);


        verticalLayout->addLayout(horizontalLayout);


        retranslateUi(AddPointDialog);
        QObject::connect(pushButton_2, SIGNAL(clicked()), AddPointDialog, SLOT(reject()));
        QObject::connect(pushButton, SIGNAL(clicked()), AddPointDialog, SLOT(accept()));

        QMetaObject::connectSlotsByName(AddPointDialog);
    } // setupUi

    void retranslateUi(QDialog *AddPointDialog)
    {
        AddPointDialog->setWindowTitle(QApplication::translate("AddPointDialog", "AddPointDialog", nullptr));
        label->setText(QApplication::translate("AddPointDialog", "X Coordinate", nullptr));
        label_2->setText(QApplication::translate("AddPointDialog", "Y Coordinate", nullptr));
        label_3->setText(QApplication::translate("AddPointDialog", "Z Coordinate", nullptr));
        label_4->setText(QApplication::translate("AddPointDialog", "Wavelength", nullptr));
        label_5->setText(QApplication::translate("AddPointDialog", "Intensity", nullptr));
        pushButton->setText(QApplication::translate("AddPointDialog", "Add Point", nullptr));
        pushButton_2->setText(QApplication::translate("AddPointDialog", "Cancel", nullptr));
    } // retranslateUi

};

namespace Ui {
    class AddPointDialog: public Ui_AddPointDialog {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_ADDPOINTDIALOG_H
