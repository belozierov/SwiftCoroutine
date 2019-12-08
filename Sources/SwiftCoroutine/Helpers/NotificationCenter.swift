//
//  NotificationCenter.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 08.12.2019.
//  Copyright © 2019 Alex Belozierov. All rights reserved.
//

import Foundation

extension NotificationCenter {
    
    func notifyOnce(name: Notification.Name?, object: Any?, queue: OperationQueue? = nil, handler: @escaping (Notification) -> Void) {
        var token: NSObjectProtocol!
        token = addObserver(forName: name, object: object, queue: queue) { [unowned self] in
            handler($0)
            token.map(self.removeObserver)
        }
    }
    
}
