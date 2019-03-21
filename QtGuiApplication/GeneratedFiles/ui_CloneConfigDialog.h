/********************************************************************************
** Form generated from reading UI file 'CloneConfigDialog.ui'
**
** Created by: Qt User Interface Compiler version 5.12.2
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_CLONECONFIGDIALOG_H
#define UI_CLONECONFIGDIALOG_H

#include <QtCore/QVariant>
#include <QtWidgets/QApplication>
#include <QtWidgets/QComboBox>
#include <QtWidgets/QDialog>
#include <QtWidgets/QGridLayout>
#include <QtWidgets/QLabel>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QSpacerItem>

QT_BEGIN_NAMESPACE

class Ui_CloneConfigDialog
{
public:
    QGridLayout *gridLayout_2;
    QGridLayout *gridLayout_3;
    QSpacerItem *verticalSpacer;
    QGridLayout *gridLayout;
    QLabel *label;
    QComboBox *comboCloneWavelength;
    QSpacerItem *verticalSpacer_2;
    QPushButton *pushButton_2;
    QPushButton *pushButton;

    void setupUi(QDialog *CloneConfigDialog)
    {
        if (CloneConfigDialog->objectName().isEmpty())
            CloneConfigDialog->setObjectName(QString::fromUtf8("CloneConfigDialog"));
        CloneConfigDialog->resize(328, 224);
        QSizePolicy sizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
        sizePolicy.setHorizontalStretch(0);
        sizePolicy.setVerticalStretch(0);
        sizePolicy.setHeightForWidth(CloneConfigDialog->sizePolicy().hasHeightForWidth());
        CloneConfigDialog->setSizePolicy(sizePolicy);
        gridLayout_2 = new QGridLayout(CloneConfigDialog);
        gridLayout_2->setSpacing(6);
        gridLayout_2->setContentsMargins(11, 11, 11, 11);
        gridLayout_2->setObjectName(QString::fromUtf8("gridLayout_2"));
        gridLayout_3 = new QGridLayout();
        gridLayout_3->setSpacing(6);
        gridLayout_3->setObjectName(QString::fromUtf8("gridLayout_3"));
        verticalSpacer = new QSpacerItem(20, 52, QSizePolicy::Minimum, QSizePolicy::Expanding);

        gridLayout_3->addItem(verticalSpacer, 0, 0, 1, 1);

        gridLayout = new QGridLayout();
        gridLayout->setSpacing(6);
        gridLayout->setObjectName(QString::fromUtf8("gridLayout"));
        label = new QLabel(CloneConfigDialog);
        label->setObjectName(QString::fromUtf8("label"));

        gridLayout->addWidget(label, 0, 0, 1, 1);

        comboCloneWavelength = new QComboBox(CloneConfigDialog);
        comboCloneWavelength->setObjectName(QString::fromUtf8("comboCloneWavelength"));
        QSizePolicy sizePolicy1(QSizePolicy::Expanding, QSizePolicy::Fixed);
        sizePolicy1.setHorizontalStretch(0);
        sizePolicy1.setVerticalStretch(0);
        sizePolicy1.setHeightForWidth(comboCloneWavelength->sizePolicy().hasHeightForWidth());
        comboCloneWavelength->setSizePolicy(sizePolicy1);

        gridLayout->addWidget(comboCloneWavelength, 1, 0, 1, 1);


        gridLayout_3->addLayout(gridLayout, 1, 0, 1, 1);

        verticalSpacer_2 = new QSpacerItem(20, 52, QSizePolicy::Minimum, QSizePolicy::Expanding);

        gridLayout_3->addItem(verticalSpacer_2, 2, 0, 1, 1);


        gridLayout_2->addLayout(gridLayout_3, 0, 0, 1, 2);

        pushButton_2 = new QPushButton(CloneConfigDialog);
        pushButton_2->setObjectName(QString::fromUtf8("pushButton_2"));

        gridLayout_2->addWidget(pushButton_2, 1, 1, 1, 1);

        pushButton = new QPushButton(CloneConfigDialog);
        pushButton->setObjectName(QString::fromUtf8("pushButton"));

        gridLayout_2->addWidget(pushButton, 1, 0, 1, 1);


        retranslateUi(CloneConfigDialog);
        QObject::connect(pushButton_2, SIGNAL(clicked()), CloneConfigDialog, SLOT(reject()));
        QObject::connect(pushButton, SIGNAL(clicked()), CloneConfigDialog, SLOT(accept()));

        pushButton->setDefault(true);


        QMetaObject::connectSlotsByName(CloneConfigDialog);
    } // setupUi

    void retranslateUi(QDialog *CloneConfigDialog)
    {
        CloneConfigDialog->setWindowTitle(QApplication::translate("CloneConfigDialog", "Clone Configuration", nullptr));
        label->setText(QApplication::translate("CloneConfigDialog", "Clone The Optical Configuration Of This Wavelength:", nullptr));
        pushButton_2->setText(QApplication::translate("CloneConfigDialog", "Cancel", nullptr));
        pushButton->setText(QApplication::translate("CloneConfigDialog", "Accept", nullptr));
    } // retranslateUi

};

namespace Ui {
    class CloneConfigDialog: public Ui_CloneConfigDialog {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_CLONECONFIGDIALOG_H
