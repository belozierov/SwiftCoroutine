//
//  CoCancellable.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 10.05.2020.
//  Copyright © 2020 Alex Belozierov. All rights reserved.
//

public protocol CoCancellable: class {
    
    /// Cancels the current `CoCancellable`.
    func cancel()
    
    /// Adds an observer callback that is called when the `CoCancellable` is completed.
    /// - Parameter callback: The callback that is called when the `CoCancellable` is completed.
    func whenComplete(_ callback: @escaping () -> Void)
    
}

extension CoCancellable {
    
    /// Adds weak referance of `self` to `CoScope`.
    /// - Parameter scope: `CoScope` to add `self` to.
    /// - Returns: The current `CoCancellable`.
    @discardableResult @inlinable
    public func added(to scope: CoScope) -> Self {
        scope.add(self)
        return self
    }
    
}

extension CoScope: CoCancellable {}
extension CoFuture: CoCancellable {}
extension CoChannel: CoCancellable {}
extension CoChannel.Receiver: CoCancellable {}
extension CoChannel.Sender: CoCancellable {}



