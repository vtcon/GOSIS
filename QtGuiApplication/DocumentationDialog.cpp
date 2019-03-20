#include "DocumentationDialog.h"

#include "QtWebEngineWidgets/qwebengineview.h"
#include <QUrl>

#include "qfileinfo.h"

DocumentationDialog::DocumentationDialog(QWidget *parent)
	: QDialog(parent)
{
	setupUi(this);
	webView->load(QUrl::fromLocalFile(QFileInfo("help/Help.html").absoluteFilePath()));
	webView->show();
}

DocumentationDialog::~DocumentationDialog()
{
}
