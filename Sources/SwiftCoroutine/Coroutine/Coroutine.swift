//
//  Coroutine.swift
//  SwiftCoroutine iOS
//
//  Created by Alex Belozierov on 08.12.2019.
//  Copyright © 2019 Alex Belozierov. All rights reserved.
//

import Foundation

public protocol Coroutine: class {
    
    typealias Block = () -> Void
    
    func resume()
    func suspend()
    func setDispatcher(_ dispatcher: AsyncCoroutine.Dispatcher)
    
}

extension Coroutine {
    
    @inline(__always) var isCurrent: Bool {
        Thread.current.currentCoroutine === self
    }
    
    // MARK: - Notifications
    
    func postSuspend(finished: Bool) {
        let name: Notification.Name = finished ? .coroutineDidComplete : .coroutineDidSuspend
        notificationCenter.post(name: name, object: self)
    }
    
    public func notifyOnceOnSuspend(handler: @escaping Block) {
        notifyOnce(name: .coroutineDidSuspend, handler: handler)
    }
    
    public func notifyOnCompletion(handler: @escaping Block) {
        notifyOnce(name: .coroutineDidComplete, handler: handler)
    }
    
    private func notifyOnce(name: Notification.Name, handler: @escaping Block) {
        notificationCenter.notifyOnce(name: name, object: self) { _ in handler() }
    }
    
    private var notificationCenter: NotificationCenter { .default }
    
}

extension Thread {
    
    public var currentCoroutine: Coroutine? {
        @inline(__always) get { threadDictionary.value(forKey: #function) as? Coroutine }
        @inline(__always) set { threadDictionary.setValue(newValue, forKey: #function) }
    }
    
}

extension Notification.Name {
    
    public static let coroutineDidSuspend = Notification.Name(#function)
    public static let coroutineDidComplete = Notification.Name(#function)
    
}
