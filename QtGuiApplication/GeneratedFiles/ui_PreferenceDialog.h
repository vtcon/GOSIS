/********************************************************************************
** Form generated from reading UI file 'PreferenceDialog.ui'
**
** Created by: Qt User Interface Compiler version 5.12.0
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_PREFERENCEDIALOG_H
#define UI_PREFERENCEDIALOG_H

#include <QtCore/QVariant>
#include <QtWidgets/QApplication>
#include <QtWidgets/QComboBox>
#include <QtWidgets/QDialog>
#include <QtWidgets/QGridLayout>
#include <QtWidgets/QGroupBox>
#include <QtWidgets/QLabel>
#include <QtWidgets/QLineEdit>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QSpacerItem>

QT_BEGIN_NAMESPACE

class Ui_PreferenceDialog
{
public:
    QGridLayout *gridLayout_5;
    QGroupBox *groupBox;
    QGridLayout *gridLayout_2;
    QLabel *label_13;
    QLineEdit *lineCPUThread;
    QLabel *label;
    QComboBox *comboThreadCount;
    QLabel *label_2;
    QLineEdit *lineRayGeneration;
    QLabel *label_3;
    QLineEdit *lineTraceSize;
    QLabel *label_4;
    QLineEdit *lineRenderSize;
    QGroupBox *groupBox_2;
    QGridLayout *gridLayout_4;
    QLabel *label_5;
    QComboBox *comboRGB;
    QLabel *label_6;
    QComboBox *comboRaw;
    QLabel *label_7;
    QComboBox *comboProjection;
    QLabel *label_8;
    QLineEdit *linePreviewSize;
    QLabel *label_9;
    QLabel *label_10;
    QLineEdit *lineWavelengthR;
    QLabel *label_11;
    QLineEdit *lineWavelengthG;
    QLabel *label_12;
    QLineEdit *lineWavelengthB;
    QGroupBox *groupBox_3;
    QGridLayout *gridLayout_3;
    QLabel *label_14;
    QLineEdit *lineTestKernelRepetition;
    QSpacerItem *verticalSpacer;
    QGridLayout *gridLayout;
    QPushButton *pushCancel;
    QSpacerItem *horizontalSpacer;
    QPushButton *pushDefault;
    QPushButton *pushOK;

    void setupUi(QDialog *PreferenceDialog)
    {
        if (PreferenceDialog->objectName().isEmpty())
            PreferenceDialog->setObjectName(QString::fromUtf8("PreferenceDialog"));
        PreferenceDialog->resize(546, 661);
        QSizePolicy sizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
        sizePolicy.setHorizontalStretch(0);
        sizePolicy.setVerticalStretch(0);
        sizePolicy.setHeightForWidth(PreferenceDialog->sizePolicy().hasHeightForWidth());
        PreferenceDialog->setSizePolicy(sizePolicy);
        gridLayout_5 = new QGridLayout(PreferenceDialog);
        gridLayout_5->setSpacing(6);
        gridLayout_5->setContentsMargins(11, 11, 11, 11);
        gridLayout_5->setObjectName(QString::fromUtf8("gridLayout_5"));
        groupBox = new QGroupBox(PreferenceDialog);
        groupBox->setObjectName(QString::fromUtf8("groupBox"));
        QSizePolicy sizePolicy1(QSizePolicy::Preferred, QSizePolicy::Expanding);
        sizePolicy1.setHorizontalStretch(0);
        sizePolicy1.setVerticalStretch(0);
        sizePolicy1.setHeightForWidth(groupBox->sizePolicy().hasHeightForWidth());
        groupBox->setSizePolicy(sizePolicy1);
        gridLayout_2 = new QGridLayout(groupBox);
        gridLayout_2->setSpacing(6);
        gridLayout_2->setContentsMargins(11, 11, 11, 11);
        gridLayout_2->setObjectName(QString::fromUtf8("gridLayout_2"));
        label_13 = new QLabel(groupBox);
        label_13->setObjectName(QString::fromUtf8("label_13"));

        gridLayout_2->addWidget(label_13, 0, 0, 1, 1);

        lineCPUThread = new QLineEdit(groupBox);
        lineCPUThread->setObjectName(QString::fromUtf8("lineCPUThread"));

        gridLayout_2->addWidget(lineCPUThread, 0, 1, 1, 1);

        label = new QLabel(groupBox);
        label->setObjectName(QString::fromUtf8("label"));
        QSizePolicy sizePolicy2(QSizePolicy::Expanding, QSizePolicy::Preferred);
        sizePolicy2.setHorizontalStretch(0);
        sizePolicy2.setVerticalStretch(0);
        sizePolicy2.setHeightForWidth(label->sizePolicy().hasHeightForWidth());
        label->setSizePolicy(sizePolicy2);

        gridLayout_2->addWidget(label, 1, 0, 1, 1);

        comboThreadCount = new QComboBox(groupBox);
        comboThreadCount->addItem(QString());
        comboThreadCount->addItem(QString());
        comboThreadCount->addItem(QString());
        comboThreadCount->addItem(QString());
        comboThreadCount->addItem(QString());
        comboThreadCount->setObjectName(QString::fromUtf8("comboThreadCount"));
        QSizePolicy sizePolicy3(QSizePolicy::Expanding, QSizePolicy::Fixed);
        sizePolicy3.setHorizontalStretch(0);
        sizePolicy3.setVerticalStretch(0);
        sizePolicy3.setHeightForWidth(comboThreadCount->sizePolicy().hasHeightForWidth());
        comboThreadCount->setSizePolicy(sizePolicy3);

        gridLayout_2->addWidget(comboThreadCount, 1, 1, 1, 1);

        label_2 = new QLabel(groupBox);
        label_2->setObjectName(QString::fromUtf8("label_2"));
        QSizePolicy sizePolicy4(QSizePolicy::Preferred, QSizePolicy::Preferred);
        sizePolicy4.setHorizontalStretch(0);
        sizePolicy4.setVerticalStretch(0);
        sizePolicy4.setHeightForWidth(label_2->sizePolicy().hasHeightForWidth());
        label_2->setSizePolicy(sizePolicy4);

        gridLayout_2->addWidget(label_2, 2, 0, 1, 1);

        lineRayGeneration = new QLineEdit(groupBox);
        lineRayGeneration->setObjectName(QString::fromUtf8("lineRayGeneration"));
        sizePolicy3.setHeightForWidth(lineRayGeneration->sizePolicy().hasHeightForWidth());
        lineRayGeneration->setSizePolicy(sizePolicy3);

        gridLayout_2->addWidget(lineRayGeneration, 2, 1, 1, 1);

        label_3 = new QLabel(groupBox);
        label_3->setObjectName(QString::fromUtf8("label_3"));
        sizePolicy4.setHeightForWidth(label_3->sizePolicy().hasHeightForWidth());
        label_3->setSizePolicy(sizePolicy4);

        gridLayout_2->addWidget(label_3, 3, 0, 1, 1);

        lineTraceSize = new QLineEdit(groupBox);
        lineTraceSize->setObjectName(QString::fromUtf8("lineTraceSize"));
        sizePolicy3.setHeightForWidth(lineTraceSize->sizePolicy().hasHeightForWidth());
        lineTraceSize->setSizePolicy(sizePolicy3);

        gridLayout_2->addWidget(lineTraceSize, 3, 1, 1, 1);

        label_4 = new QLabel(groupBox);
        label_4->setObjectName(QString::fromUtf8("label_4"));
        sizePolicy4.setHeightForWidth(label_4->sizePolicy().hasHeightForWidth());
        label_4->setSizePolicy(sizePolicy4);

        gridLayout_2->addWidget(label_4, 4, 0, 1, 1);

        lineRenderSize = new QLineEdit(groupBox);
        lineRenderSize->setObjectName(QString::fromUtf8("lineRenderSize"));
        sizePolicy3.setHeightForWidth(lineRenderSize->sizePolicy().hasHeightForWidth());
        lineRenderSize->setSizePolicy(sizePolicy3);

        gridLayout_2->addWidget(lineRenderSize, 4, 1, 1, 1);


        gridLayout_5->addWidget(groupBox, 0, 0, 1, 1);

        groupBox_2 = new QGroupBox(PreferenceDialog);
        groupBox_2->setObjectName(QString::fromUtf8("groupBox_2"));
        QSizePolicy sizePolicy5(QSizePolicy::Preferred, QSizePolicy::Fixed);
        sizePolicy5.setHorizontalStretch(0);
        sizePolicy5.setVerticalStretch(0);
        sizePolicy5.setHeightForWidth(groupBox_2->sizePolicy().hasHeightForWidth());
        groupBox_2->setSizePolicy(sizePolicy5);
        gridLayout_4 = new QGridLayout(groupBox_2);
        gridLayout_4->setSpacing(6);
        gridLayout_4->setContentsMargins(11, 11, 11, 11);
        gridLayout_4->setObjectName(QString::fromUtf8("gridLayout_4"));
        label_5 = new QLabel(groupBox_2);
        label_5->setObjectName(QString::fromUtf8("label_5"));
        sizePolicy4.setHeightForWidth(label_5->sizePolicy().hasHeightForWidth());
        label_5->setSizePolicy(sizePolicy4);

        gridLayout_4->addWidget(label_5, 0, 0, 1, 1);

        comboRGB = new QComboBox(groupBox_2);
        comboRGB->addItem(QString());
        comboRGB->addItem(QString());
        comboRGB->setObjectName(QString::fromUtf8("comboRGB"));

        gridLayout_4->addWidget(comboRGB, 0, 1, 1, 1);

        label_6 = new QLabel(groupBox_2);
        label_6->setObjectName(QString::fromUtf8("label_6"));
        label_6->setWordWrap(true);

        gridLayout_4->addWidget(label_6, 1, 0, 1, 1);

        comboRaw = new QComboBox(groupBox_2);
        comboRaw->addItem(QString());
        comboRaw->addItem(QString());
        comboRaw->setObjectName(QString::fromUtf8("comboRaw"));

        gridLayout_4->addWidget(comboRaw, 1, 1, 1, 1);

        label_7 = new QLabel(groupBox_2);
        label_7->setObjectName(QString::fromUtf8("label_7"));

        gridLayout_4->addWidget(label_7, 2, 0, 1, 1);

        comboProjection = new QComboBox(groupBox_2);
        comboProjection->addItem(QString());
        comboProjection->addItem(QString());
        comboProjection->addItem(QString());
        comboProjection->setObjectName(QString::fromUtf8("comboProjection"));

        gridLayout_4->addWidget(comboProjection, 2, 1, 1, 1);

        label_8 = new QLabel(groupBox_2);
        label_8->setObjectName(QString::fromUtf8("label_8"));

        gridLayout_4->addWidget(label_8, 3, 0, 1, 1);

        linePreviewSize = new QLineEdit(groupBox_2);
        linePreviewSize->setObjectName(QString::fromUtf8("linePreviewSize"));

        gridLayout_4->addWidget(linePreviewSize, 3, 1, 1, 1);

        label_9 = new QLabel(groupBox_2);
        label_9->setObjectName(QString::fromUtf8("label_9"));

        gridLayout_4->addWidget(label_9, 4, 0, 1, 2);

        label_10 = new QLabel(groupBox_2);
        label_10->setObjectName(QString::fromUtf8("label_10"));

        gridLayout_4->addWidget(label_10, 5, 0, 1, 1);

        lineWavelengthR = new QLineEdit(groupBox_2);
        lineWavelengthR->setObjectName(QString::fromUtf8("lineWavelengthR"));

        gridLayout_4->addWidget(lineWavelengthR, 5, 1, 1, 1);

        label_11 = new QLabel(groupBox_2);
        label_11->setObjectName(QString::fromUtf8("label_11"));

        gridLayout_4->addWidget(label_11, 6, 0, 1, 1);

        lineWavelengthG = new QLineEdit(groupBox_2);
        lineWavelengthG->setObjectName(QString::fromUtf8("lineWavelengthG"));

        gridLayout_4->addWidget(lineWavelengthG, 6, 1, 1, 1);

        label_12 = new QLabel(groupBox_2);
        label_12->setObjectName(QString::fromUtf8("label_12"));

        gridLayout_4->addWidget(label_12, 7, 0, 1, 1);

        lineWavelengthB = new QLineEdit(groupBox_2);
        lineWavelengthB->setObjectName(QString::fromUtf8("lineWavelengthB"));

        gridLayout_4->addWidget(lineWavelengthB, 7, 1, 1, 1);


        gridLayout_5->addWidget(groupBox_2, 1, 0, 1, 1);

        groupBox_3 = new QGroupBox(PreferenceDialog);
        groupBox_3->setObjectName(QString::fromUtf8("groupBox_3"));
        gridLayout_3 = new QGridLayout(groupBox_3);
        gridLayout_3->setSpacing(6);
        gridLayout_3->setContentsMargins(11, 11, 11, 11);
        gridLayout_3->setObjectName(QString::fromUtf8("gridLayout_3"));
        label_14 = new QLabel(groupBox_3);
        label_14->setObjectName(QString::fromUtf8("label_14"));
        sizePolicy2.setHeightForWidth(label_14->sizePolicy().hasHeightForWidth());
        label_14->setSizePolicy(sizePolicy2);

        gridLayout_3->addWidget(label_14, 0, 0, 1, 1);

        lineTestKernelRepetition = new QLineEdit(groupBox_3);
        lineTestKernelRepetition->setObjectName(QString::fromUtf8("lineTestKernelRepetition"));
        sizePolicy5.setHeightForWidth(lineTestKernelRepetition->sizePolicy().hasHeightForWidth());
        lineTestKernelRepetition->setSizePolicy(sizePolicy5);

        gridLayout_3->addWidget(lineTestKernelRepetition, 0, 1, 1, 1);


        gridLayout_5->addWidget(groupBox_3, 2, 0, 1, 1);

        verticalSpacer = new QSpacerItem(20, 54, QSizePolicy::Minimum, QSizePolicy::Expanding);

        gridLayout_5->addItem(verticalSpacer, 3, 0, 1, 1);

        gridLayout = new QGridLayout();
        gridLayout->setSpacing(6);
        gridLayout->setObjectName(QString::fromUtf8("gridLayout"));
        pushCancel = new QPushButton(PreferenceDialog);
        pushCancel->setObjectName(QString::fromUtf8("pushCancel"));

        gridLayout->addWidget(pushCancel, 0, 3, 1, 1);

        horizontalSpacer = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        gridLayout->addItem(horizontalSpacer, 0, 0, 1, 1);

        pushDefault = new QPushButton(PreferenceDialog);
        pushDefault->setObjectName(QString::fromUtf8("pushDefault"));

        gridLayout->addWidget(pushDefault, 0, 2, 1, 1);

        pushOK = new QPushButton(PreferenceDialog);
        pushOK->setObjectName(QString::fromUtf8("pushOK"));

        gridLayout->addWidget(pushOK, 0, 1, 1, 1);


        gridLayout_5->addLayout(gridLayout, 4, 0, 1, 1);


        retranslateUi(PreferenceDialog);
        QObject::connect(pushCancel, SIGNAL(clicked()), PreferenceDialog, SLOT(reject()));

        QMetaObject::connectSlotsByName(PreferenceDialog);
    } // setupUi

    void retranslateUi(QDialog *PreferenceDialog)
    {
        PreferenceDialog->setWindowTitle(QApplication::translate("PreferenceDialog", "Preferences", nullptr));
        groupBox->setTitle(QApplication::translate("PreferenceDialog", "Tracing and Rendering", nullptr));
        label_13->setText(QApplication::translate("PreferenceDialog", "Maximum CPU Threads", nullptr));
        lineCPUThread->setText(QApplication::translate("PreferenceDialog", "10", nullptr));
        label->setText(QApplication::translate("PreferenceDialog", "GPU Threads Per Kernel Launch", nullptr));
        comboThreadCount->setItemText(0, QApplication::translate("PreferenceDialog", "32", nullptr));
        comboThreadCount->setItemText(1, QApplication::translate("PreferenceDialog", "16", nullptr));
        comboThreadCount->setItemText(2, QApplication::translate("PreferenceDialog", "8", nullptr));
        comboThreadCount->setItemText(3, QApplication::translate("PreferenceDialog", "64", nullptr));
        comboThreadCount->setItemText(4, QApplication::translate("PreferenceDialog", "128", nullptr));

        label_2->setText(QApplication::translate("PreferenceDialog", "Linear Ray Generation Density", nullptr));
        lineRayGeneration->setText(QApplication::translate("PreferenceDialog", "25", nullptr));
        label_3->setText(QApplication::translate("PreferenceDialog", "Points Traced per Kernel Launch", nullptr));
        lineTraceSize->setText(QApplication::translate("PreferenceDialog", "10", nullptr));
        label_4->setText(QApplication::translate("PreferenceDialog", "Points Rendered per Kernel Launch", nullptr));
        lineRenderSize->setText(QApplication::translate("PreferenceDialog", "10", nullptr));
        groupBox_2->setTitle(QApplication::translate("PreferenceDialog", "Image and Output", nullptr));
        label_5->setText(QApplication::translate("PreferenceDialog", "RGB Standard", nullptr));
        comboRGB->setItemText(0, QApplication::translate("PreferenceDialog", "AdobeRGB", nullptr));
        comboRGB->setItemText(1, QApplication::translate("PreferenceDialog", "sRGB", nullptr));

        label_6->setText(QApplication::translate("PreferenceDialog", "Color Matching Function for Raw Data", nullptr));
        comboRaw->setItemText(0, QApplication::translate("PreferenceDialog", "XYZ (CIE 2006)", nullptr));
        comboRaw->setItemText(1, QApplication::translate("PreferenceDialog", "LMS (CIE 2006)", nullptr));

        label_7->setText(QApplication::translate("PreferenceDialog", "Image Projection Method (from sphere to plane)", nullptr));
        comboProjection->setItemText(0, QApplication::translate("PreferenceDialog", "Plate Carree", nullptr));
        comboProjection->setItemText(1, QApplication::translate("PreferenceDialog", "None", nullptr));
        comboProjection->setItemText(2, QApplication::translate("PreferenceDialog", "Along Z-axis", nullptr));

        label_8->setText(QApplication::translate("PreferenceDialog", "Preview Window Size (px)", nullptr));
        linePreviewSize->setText(QApplication::translate("PreferenceDialog", "800", nullptr));
        label_9->setText(QApplication::translate("PreferenceDialog", "Primary Wavelengths for RGB image Reconstruction:", nullptr));
        label_10->setText(QApplication::translate("PreferenceDialog", "Long Wavelength (Red)", nullptr));
        lineWavelengthR->setText(QApplication::translate("PreferenceDialog", "620", nullptr));
        label_11->setText(QApplication::translate("PreferenceDialog", "Medium Wavelength (Green)", nullptr));
        lineWavelengthG->setText(QApplication::translate("PreferenceDialog", "530", nullptr));
        label_12->setText(QApplication::translate("PreferenceDialog", "Short Wavelength (Blue)", nullptr));
        lineWavelengthB->setText(QApplication::translate("PreferenceDialog", "465", nullptr));
        groupBox_3->setTitle(QApplication::translate("PreferenceDialog", "Testing and Benchmarking", nullptr));
        label_14->setText(QApplication::translate("PreferenceDialog", "Test Kernel Repetition", nullptr));
        lineTestKernelRepetition->setText(QApplication::translate("PreferenceDialog", "10000", nullptr));
        pushCancel->setText(QApplication::translate("PreferenceDialog", "Cancel", nullptr));
        pushDefault->setText(QApplication::translate("PreferenceDialog", "Default", nullptr));
        pushOK->setText(QApplication::translate("PreferenceDialog", "OK", nullptr));
    } // retranslateUi

};

namespace Ui {
    class PreferenceDialog: public Ui_PreferenceDialog {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_PREFERENCEDIALOG_H
