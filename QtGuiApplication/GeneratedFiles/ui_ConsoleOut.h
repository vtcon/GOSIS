/********************************************************************************
** Form generated from reading UI file 'ConsoleOut.ui'
**
** Created by: Qt User Interface Compiler version 5.12.2
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_CONSOLEOUT_H
#define UI_CONSOLEOUT_H

#include <QtCore/QVariant>
#include <QtWidgets/QApplication>
#include <QtWidgets/QGridLayout>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QSpacerItem>
#include <QtWidgets/QTextEdit>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_ConsoleOut
{
public:
    QGridLayout *gridLayout;
    QTextEdit *textEdit;
    QSpacerItem *horizontalSpacer;
    QPushButton *pushClear;

    void setupUi(QWidget *ConsoleOut)
    {
        if (ConsoleOut->objectName().isEmpty())
            ConsoleOut->setObjectName(QString::fromUtf8("ConsoleOut"));
        ConsoleOut->resize(785, 604);
        gridLayout = new QGridLayout(ConsoleOut);
        gridLayout->setSpacing(6);
        gridLayout->setContentsMargins(11, 11, 11, 11);
        gridLayout->setObjectName(QString::fromUtf8("gridLayout"));
        textEdit = new QTextEdit(ConsoleOut);
        textEdit->setObjectName(QString::fromUtf8("textEdit"));

        gridLayout->addWidget(textEdit, 0, 0, 1, 2);

        horizontalSpacer = new QSpacerItem(660, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        gridLayout->addItem(horizontalSpacer, 1, 0, 1, 1);

        pushClear = new QPushButton(ConsoleOut);
        pushClear->setObjectName(QString::fromUtf8("pushClear"));

        gridLayout->addWidget(pushClear, 1, 1, 1, 1);


        retranslateUi(ConsoleOut);
        QObject::connect(pushClear, SIGNAL(clicked()), textEdit, SLOT(clear()));

        QMetaObject::connectSlotsByName(ConsoleOut);
    } // setupUi

    void retranslateUi(QWidget *ConsoleOut)
    {
        ConsoleOut->setWindowTitle(QApplication::translate("ConsoleOut", "ConsoleOut", nullptr));
        pushClear->setText(QApplication::translate("ConsoleOut", "Clear", nullptr));
    } // retranslateUi

};

namespace Ui {
    class ConsoleOut: public Ui_ConsoleOut {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_CONSOLEOUT_H
