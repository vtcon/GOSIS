/********************************************************************************
** Form generated from reading UI file 'AddPointDialog.ui'
**
** Created by: Qt User Interface Compiler version 5.12.2
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

QT_BEGIN_NAMESPACE

class Ui_AddPointDialog
{
public:
    QGridLayout *gridLayout_2;
    QSpacerItem *verticalSpacer;
    QGridLayout *gridLayout;
    QLabel *label;
    QLineEdit *lineWavelength;
    QLabel *label_2;
    QLineEdit *lineX;
    QLabel *label_3;
    QLineEdit *lineY;
    QLabel *label_4;
    QLineEdit *lineZ;
    QLabel *label_5;
    QLineEdit *lineIntensity;
    QSpacerItem *verticalSpacer_2;
    QHBoxLayout *horizontalLayout;
    QSpacerItem *horizontalSpacer;
    QPushButton *pushButton;
    QPushButton *pushButton_2;

    void setupUi(QDialog *AddPointDialog)
    {
        if (AddPointDialog->objectName().isEmpty())
            AddPointDialog->setObjectName(QString::fromUtf8("AddPointDialog"));
        AddPointDialog->resize(400, 300);
        gridLayout_2 = new QGridLayout(AddPointDialog);
        gridLayout_2->setSpacing(6);
        gridLayout_2->setContentsMargins(11, 11, 11, 11);
        gridLayout_2->setObjectName(QString::fromUtf8("gridLayout_2"));
        verticalSpacer = new QSpacerItem(20, 35, QSizePolicy::Minimum, QSizePolicy::Expanding);

        gridLayout_2->addItem(verticalSpacer, 0, 0, 1, 1);

        gridLayout = new QGridLayout();
        gridLayout->setSpacing(6);
        gridLayout->setObjectName(QString::fromUtf8("gridLayout"));
        label = new QLabel(AddPointDialog);
        label->setObjectName(QString::fromUtf8("label"));

        gridLayout->addWidget(label, 0, 0, 1, 1);

        lineWavelength = new QLineEdit(AddPointDialog);
        lineWavelength->setObjectName(QString::fromUtf8("lineWavelength"));

        gridLayout->addWidget(lineWavelength, 0, 1, 1, 1);

        label_2 = new QLabel(AddPointDialog);
        label_2->setObjectName(QString::fromUtf8("label_2"));

        gridLayout->addWidget(label_2, 1, 0, 1, 1);

        lineX = new QLineEdit(AddPointDialog);
        lineX->setObjectName(QString::fromUtf8("lineX"));

        gridLayout->addWidget(lineX, 1, 1, 1, 1);

        label_3 = new QLabel(AddPointDialog);
        label_3->setObjectName(QString::fromUtf8("label_3"));

        gridLayout->addWidget(label_3, 2, 0, 1, 1);

        lineY = new QLineEdit(AddPointDialog);
        lineY->setObjectName(QString::fromUtf8("lineY"));

        gridLayout->addWidget(lineY, 2, 1, 1, 1);

        label_4 = new QLabel(AddPointDialog);
        label_4->setObjectName(QString::fromUtf8("label_4"));

        gridLayout->addWidget(label_4, 3, 0, 1, 1);

        lineZ = new QLineEdit(AddPointDialog);
        lineZ->setObjectName(QString::fromUtf8("lineZ"));

        gridLayout->addWidget(lineZ, 3, 1, 1, 1);

        label_5 = new QLabel(AddPointDialog);
        label_5->setObjectName(QString::fromUtf8("label_5"));

        gridLayout->addWidget(label_5, 4, 0, 1, 1);

        lineIntensity = new QLineEdit(AddPointDialog);
        lineIntensity->setObjectName(QString::fromUtf8("lineIntensity"));

        gridLayout->addWidget(lineIntensity, 4, 1, 1, 1);


        gridLayout_2->addLayout(gridLayout, 1, 0, 1, 1);

        verticalSpacer_2 = new QSpacerItem(20, 36, QSizePolicy::Minimum, QSizePolicy::Expanding);

        gridLayout_2->addItem(verticalSpacer_2, 2, 0, 1, 1);

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


        gridLayout_2->addLayout(horizontalLayout, 3, 0, 1, 1);


        retranslateUi(AddPointDialog);
        QObject::connect(pushButton_2, SIGNAL(clicked()), AddPointDialog, SLOT(reject()));

        QMetaObject::connectSlotsByName(AddPointDialog);
    } // setupUi

    void retranslateUi(QDialog *AddPointDialog)
    {
        AddPointDialog->setWindowTitle(QApplication::translate("AddPointDialog", "AddPointDialog", nullptr));
        label->setText(QApplication::translate("AddPointDialog", "Wavelength", nullptr));
        lineWavelength->setText(QApplication::translate("AddPointDialog", "555.0", nullptr));
        label_2->setText(QApplication::translate("AddPointDialog", "X Coordinate", nullptr));
        lineX->setText(QApplication::translate("AddPointDialog", "0.0", nullptr));
        label_3->setText(QApplication::translate("AddPointDialog", "Y Coordinate", nullptr));
        lineY->setText(QApplication::translate("AddPointDialog", "0.0", nullptr));
        label_4->setText(QApplication::translate("AddPointDialog", "Z Coordinate", nullptr));
        lineZ->setText(QApplication::translate("AddPointDialog", "250.0", nullptr));
        label_5->setText(QApplication::translate("AddPointDialog", "Intensity", nullptr));
        lineIntensity->setText(QApplication::translate("AddPointDialog", "1.0", nullptr));
        pushButton->setText(QApplication::translate("AddPointDialog", "Add", nullptr));
        pushButton_2->setText(QApplication::translate("AddPointDialog", "Cancel", nullptr));
    } // retranslateUi

};

namespace Ui {
    class AddPointDialog: public Ui_AddPointDialog {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_ADDPOINTDIALOG_H
