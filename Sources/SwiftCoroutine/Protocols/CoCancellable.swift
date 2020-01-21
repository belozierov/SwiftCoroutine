//
//  CoCancellable.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 04.01.2020.
//  Copyright © 2020 Alex Belozierov. All rights reserved.
//

import Combine

public protocol CoCancellable: Cancellable, Hashable {
    
    func cancel()
    func cancelUpstream()
    
}

extension CoCancellable where Self: AnyObject {
    
    @inlinable var anyCoCancellable: AnyCoCancellable {
        AnyCoCancellable(self)
    }
    
}

final public class AnyCoCancellable: CoCancellable {
    
    private let _cancel, _cancelUpstream: () -> Void
    
    public init<T: CoCancellable & AnyObject>(_ cancellable: T) {
        _cancel = { [weak cancellable] in cancellable?.cancel() }
        _cancelUpstream = { [weak cancellable] in cancellable?.cancelUpstream() }
    }
    
    public func cancel() {
        _cancel()
    }
    
    public func cancelUpstream() {
        _cancelUpstream()
    }
    
    @inlinable public static func == (lhs: AnyCoCancellable, rhs: AnyCoCancellable) -> Bool {
        lhs === rhs
    }
    
    @inlinable public func hash(into hasher: inout Hasher) {
        ObjectIdentifier(self).hash(into: &hasher)
    }
    
}
