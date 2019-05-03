/********************************************************************************
** Form generated from reading UI file 'AddSurfaceDialog.ui'
**
** Created by: Qt User Interface Compiler version 5.12.2
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_ADDSURFACEDIALOG_H
#define UI_ADDSURFACEDIALOG_H

#include <QtCore/QVariant>
#include <QtWidgets/QApplication>
#include <QtWidgets/QComboBox>
#include <QtWidgets/QDialog>
#include <QtWidgets/QGridLayout>
#include <QtWidgets/QLabel>
#include <QtWidgets/QLineEdit>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QSpacerItem>

QT_BEGIN_NAMESPACE

class Ui_AddSurfaceDialog
{
public:
    QGridLayout *gridLayout_2;
    QSpacerItem *verticalSpacer;
    QGridLayout *gridLayout;
    QLabel *label;
    QLineEdit *lineZ;
    QLabel *label_2;
    QLineEdit *lineX;
    QLabel *label_3;
    QLineEdit *lineY;
    QLabel *label_4;
    QLineEdit *lineDiam;
    QLabel *label_5;
    QLineEdit *lineRadius;
    QLabel *label_6;
    QLineEdit *lineRefracI;
    QLabel *label_7;
    QLineEdit *lineAsph;
    QLabel *label_8;
    QComboBox *comboBox;
    QPushButton *pushSelectApoPath;
    QLineEdit *lineApoPath;
    QSpacerItem *verticalSpacer_2;
    QSpacerItem *horizontalSpacer_2;
    QPushButton *pushAddSurface;
    QPushButton *pushButton_2;

    void setupUi(QDialog *AddSurfaceDialog)
    {
        if (AddSurfaceDialog->objectName().isEmpty())
            AddSurfaceDialog->setObjectName(QString::fromUtf8("AddSurfaceDialog"));
        AddSurfaceDialog->resize(517, 369);
        gridLayout_2 = new QGridLayout(AddSurfaceDialog);
        gridLayout_2->setSpacing(6);
        gridLayout_2->setContentsMargins(11, 11, 11, 11);
        gridLayout_2->setObjectName(QString::fromUtf8("gridLayout_2"));
        verticalSpacer = new QSpacerItem(20, 25, QSizePolicy::Minimum, QSizePolicy::Expanding);

        gridLayout_2->addItem(verticalSpacer, 0, 0, 1, 1);

        gridLayout = new QGridLayout();
        gridLayout->setSpacing(6);
        gridLayout->setObjectName(QString::fromUtf8("gridLayout"));
        label = new QLabel(AddSurfaceDialog);
        label->setObjectName(QString::fromUtf8("label"));

        gridLayout->addWidget(label, 0, 0, 1, 1);

        lineZ = new QLineEdit(AddSurfaceDialog);
        lineZ->setObjectName(QString::fromUtf8("lineZ"));

        gridLayout->addWidget(lineZ, 0, 1, 1, 3);

        label_2 = new QLabel(AddSurfaceDialog);
        label_2->setObjectName(QString::fromUtf8("label_2"));

        gridLayout->addWidget(label_2, 1, 0, 1, 1);

        lineX = new QLineEdit(AddSurfaceDialog);
        lineX->setObjectName(QString::fromUtf8("lineX"));

        gridLayout->addWidget(lineX, 1, 1, 1, 3);

        label_3 = new QLabel(AddSurfaceDialog);
        label_3->setObjectName(QString::fromUtf8("label_3"));

        gridLayout->addWidget(label_3, 2, 0, 1, 1);

        lineY = new QLineEdit(AddSurfaceDialog);
        lineY->setObjectName(QString::fromUtf8("lineY"));

        gridLayout->addWidget(lineY, 2, 1, 1, 3);

        label_4 = new QLabel(AddSurfaceDialog);
        label_4->setObjectName(QString::fromUtf8("label_4"));

        gridLayout->addWidget(label_4, 3, 0, 1, 1);

        lineDiam = new QLineEdit(AddSurfaceDialog);
        lineDiam->setObjectName(QString::fromUtf8("lineDiam"));

        gridLayout->addWidget(lineDiam, 3, 1, 1, 3);

        label_5 = new QLabel(AddSurfaceDialog);
        label_5->setObjectName(QString::fromUtf8("label_5"));

        gridLayout->addWidget(label_5, 4, 0, 1, 1);

        lineRadius = new QLineEdit(AddSurfaceDialog);
        lineRadius->setObjectName(QString::fromUtf8("lineRadius"));

        gridLayout->addWidget(lineRadius, 4, 1, 1, 3);

        label_6 = new QLabel(AddSurfaceDialog);
        label_6->setObjectName(QString::fromUtf8("label_6"));

        gridLayout->addWidget(label_6, 5, 0, 1, 1);

        lineRefracI = new QLineEdit(AddSurfaceDialog);
        lineRefracI->setObjectName(QString::fromUtf8("lineRefracI"));

        gridLayout->addWidget(lineRefracI, 5, 1, 1, 3);

        label_7 = new QLabel(AddSurfaceDialog);
        label_7->setObjectName(QString::fromUtf8("label_7"));

        gridLayout->addWidget(label_7, 6, 0, 1, 1);

        lineAsph = new QLineEdit(AddSurfaceDialog);
        lineAsph->setObjectName(QString::fromUtf8("lineAsph"));

        gridLayout->addWidget(lineAsph, 6, 1, 1, 3);

        label_8 = new QLabel(AddSurfaceDialog);
        label_8->setObjectName(QString::fromUtf8("label_8"));

        gridLayout->addWidget(label_8, 7, 0, 1, 1);

        comboBox = new QComboBox(AddSurfaceDialog);
        comboBox->addItem(QString());
        comboBox->addItem(QString());
        comboBox->addItem(QString());
        comboBox->setObjectName(QString::fromUtf8("comboBox"));

        gridLayout->addWidget(comboBox, 7, 1, 1, 1);

        pushSelectApoPath = new QPushButton(AddSurfaceDialog);
        pushSelectApoPath->setObjectName(QString::fromUtf8("pushSelectApoPath"));

        gridLayout->addWidget(pushSelectApoPath, 7, 2, 1, 1);

        lineApoPath = new QLineEdit(AddSurfaceDialog);
        lineApoPath->setObjectName(QString::fromUtf8("lineApoPath"));

        gridLayout->addWidget(lineApoPath, 7, 3, 1, 1);


        gridLayout_2->addLayout(gridLayout, 1, 0, 1, 3);

        verticalSpacer_2 = new QSpacerItem(20, 26, QSizePolicy::Minimum, QSizePolicy::Expanding);

        gridLayout_2->addItem(verticalSpacer_2, 2, 0, 1, 1);

        horizontalSpacer_2 = new QSpacerItem(292, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        gridLayout_2->addItem(horizontalSpacer_2, 3, 0, 1, 1);

        pushAddSurface = new QPushButton(AddSurfaceDialog);
        pushAddSurface->setObjectName(QString::fromUtf8("pushAddSurface"));

        gridLayout_2->addWidget(pushAddSurface, 3, 1, 1, 1);

        pushButton_2 = new QPushButton(AddSurfaceDialog);
        pushButton_2->setObjectName(QString::fromUtf8("pushButton_2"));

        gridLayout_2->addWidget(pushButton_2, 3, 2, 1, 1);


        retranslateUi(AddSurfaceDialog);
        QObject::connect(pushButton_2, SIGNAL(clicked()), AddSurfaceDialog, SLOT(reject()));

        QMetaObject::connectSlotsByName(AddSurfaceDialog);
    } // setupUi

    void retranslateUi(QDialog *AddSurfaceDialog)
    {
        AddSurfaceDialog->setWindowTitle(QApplication::translate("AddSurfaceDialog", "AddSurfaceDialog", nullptr));
        label->setText(QApplication::translate("AddSurfaceDialog", "Z Coordinate", nullptr));
        lineZ->setText(QApplication::translate("AddSurfaceDialog", "100", nullptr));
        label_2->setText(QApplication::translate("AddSurfaceDialog", "X Coordinate", nullptr));
        lineX->setText(QApplication::translate("AddSurfaceDialog", "0", nullptr));
        label_3->setText(QApplication::translate("AddSurfaceDialog", "Y Coordinate", nullptr));
        lineY->setText(QApplication::translate("AddSurfaceDialog", "0", nullptr));
        label_4->setText(QApplication::translate("AddSurfaceDialog", "Diameter", nullptr));
        lineDiam->setText(QApplication::translate("AddSurfaceDialog", "40", nullptr));
        label_5->setText(QApplication::translate("AddSurfaceDialog", "Curvature Radius", nullptr));
        lineRadius->setText(QApplication::translate("AddSurfaceDialog", "26.612", nullptr));
        label_6->setText(QApplication::translate("AddSurfaceDialog", "Refractive Index", nullptr));
        lineRefracI->setText(QApplication::translate("AddSurfaceDialog", "1.5168", nullptr));
        label_7->setText(QApplication::translate("AddSurfaceDialog", "Asphericity", nullptr));
        lineAsph->setText(QApplication::translate("AddSurfaceDialog", "0.0", nullptr));
        label_8->setText(QApplication::translate("AddSurfaceDialog", "Apodization", nullptr));
        comboBox->setItemText(0, QApplication::translate("AddSurfaceDialog", "Uniform", nullptr));
        comboBox->setItemText(1, QApplication::translate("AddSurfaceDialog", "Bartlett", nullptr));
        comboBox->setItemText(2, QApplication::translate("AddSurfaceDialog", "Custom Apodization", nullptr));

        pushSelectApoPath->setText(QApplication::translate("AddSurfaceDialog", "Select File", nullptr));
        lineApoPath->setText(QString());
        lineApoPath->setPlaceholderText(QApplication::translate("AddSurfaceDialog", "Custom Apo File Path", nullptr));
        pushAddSurface->setText(QApplication::translate("AddSurfaceDialog", "Add", nullptr));
        pushButton_2->setText(QApplication::translate("AddSurfaceDialog", "Cancel", nullptr));
    } // retranslateUi

};

namespace Ui {
    class AddSurfaceDialog: public Ui_AddSurfaceDialog {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_ADDSURFACEDIALOG_H
