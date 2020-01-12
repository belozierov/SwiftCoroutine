//
//  CoHandleFutureTests.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 11.01.2020.
//  Copyright © 2020 Alex Belozierov. All rights reserved.
//

import XCTest
@testable import SwiftCoroutine

class CoHandleFutureTests: CoFutureTests {
    
    override var future: CoHandleFuture<Bool> {
        futures.handle
    }
    
    private var futures: (parent: CoFuture<Bool>, handle: CoHandleFuture<Bool>) {
        let parent = CoFuture<Bool>()
        return (parent, CoHandleFuture(parent: parent) { _ in _ = parent })
    }
    
    func testInit() {
        let (parent, handle) = futures
        XCTAssertEqual(handle.subscriptions?.count, 1)
        XCTAssertEqual(parent.subscriptions?.count, 1)
        XCTAssertEqual(parent.mutex, handle.mutex)
        XCTAssert(parent.$resultStorage.projectedValue
            === handle.$resultStorage.projectedValue)
        XCTAssertNil(handle.result)
    }
    
    func testInitWithResult() {
        let parent = CoFuture<Int>()
        parent.complete(with: .success(1))
        let handle = CoHandleFuture(parent: parent) { _ in }
        XCTAssertEqual(parent.subscriptions?.count, 0)
        XCTAssertEqual(handle.subscriptions?.count, 0)
        XCTAssertEqual(handle.result, 1)
    }
    
    override func testCancel() {
        super.testCancel()
        let (parent, handle) = futures
        handle.cancel()
        XCTAssertNil(parent.subscriptions?[handle])
    }
    
}
