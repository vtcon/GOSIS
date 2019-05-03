/********************************************************************************
** Form generated from reading UI file 'DocumentationDialog.ui'
**
** Created by: Qt User Interface Compiler version 5.12.2
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_DOCUMENTATIONDIALOG_H
#define UI_DOCUMENTATIONDIALOG_H

#include <QtCore/QVariant>
#include <QtWebEngineWidgets/QWebEngineView>
#include <QtWidgets/QApplication>
#include <QtWidgets/QDialog>
#include <QtWidgets/QGridLayout>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QSpacerItem>

QT_BEGIN_NAMESPACE

class Ui_DocumentationDialog
{
public:
    QGridLayout *gridLayout;
    QWebEngineView *webView;
    QSpacerItem *horizontalSpacer;
    QPushButton *pushButton;

    void setupUi(QDialog *DocumentationDialog)
    {
        if (DocumentationDialog->objectName().isEmpty())
            DocumentationDialog->setObjectName(QString::fromUtf8("DocumentationDialog"));
        DocumentationDialog->resize(748, 538);
        gridLayout = new QGridLayout(DocumentationDialog);
        gridLayout->setSpacing(6);
        gridLayout->setContentsMargins(11, 11, 11, 11);
        gridLayout->setObjectName(QString::fromUtf8("gridLayout"));
        webView = new QWebEngineView(DocumentationDialog);
        webView->setObjectName(QString::fromUtf8("webView"));
        webView->setUrl(QUrl(QString::fromUtf8("about:blank")));

        gridLayout->addWidget(webView, 0, 0, 1, 2);

        horizontalSpacer = new QSpacerItem(623, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        gridLayout->addItem(horizontalSpacer, 1, 0, 1, 1);

        pushButton = new QPushButton(DocumentationDialog);
        pushButton->setObjectName(QString::fromUtf8("pushButton"));

        gridLayout->addWidget(pushButton, 1, 1, 1, 1);


        retranslateUi(DocumentationDialog);
        QObject::connect(pushButton, SIGNAL(clicked()), DocumentationDialog, SLOT(accept()));

        QMetaObject::connectSlotsByName(DocumentationDialog);
    } // setupUi

    void retranslateUi(QDialog *DocumentationDialog)
    {
        DocumentationDialog->setWindowTitle(QApplication::translate("DocumentationDialog", "Documentation", nullptr));
        pushButton->setText(QApplication::translate("DocumentationDialog", "Close", nullptr));
    } // retranslateUi

};

namespace Ui {
    class DocumentationDialog: public Ui_DocumentationDialog {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_DOCUMENTATIONDIALOG_H
