#pragma once
#include <qobject.h>
class RenderProgressEmittor :
	public QObject
{
	Q_OBJECT

public:
	RenderProgressEmittor();
	~RenderProgressEmittor();
	void setProgress(int value);

public slots:
	void watchProgress();

signals:
	void progressChanges(int newvalue);

private:
	int m_progress;
};

